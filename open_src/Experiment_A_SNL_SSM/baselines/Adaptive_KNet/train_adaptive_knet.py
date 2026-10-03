# -*- coding: utf-8 -*-
"""Train Adaptive-KNet and estimate test-time SoW from innovations."""

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dynamics import AcceleratedMovementModel
from adaptive_knet_mnet import KalmanNetNN
from baselines.current_dataset import (
    acc_at,
    default_test_paths,
    flatten_dataset,
    grouped_conditions,
    load_raw_dataset,
    print_metrics,
    save_results,
    summarize_estimates,
)
from hypernetwork import HyperNetwork


class KNetSysModel:
    def __init__(self, f, h, m, n, prior_Q, prior_Sigma, prior_S):
        self.f = f
        self.h = h
        self.m = m
        self.n = n
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measurement_from_state(x):
    x2 = x.squeeze(-1)
    radius = torch.sqrt(x2[:, :3].square().sum(dim=1) + 1e-8)
    return torch.stack(
        [radius, x2[:, 3], x2[:, 4], x2[:, 5]], dim=-1
    ).unsqueeze(-1)


def measurement_jacobian(x):
    """Jacobian of [range, vx, vy, vz] for states shaped [..., 6]."""
    radius = torch.sqrt(x[..., :3].square().sum(dim=-1) + 1e-8)
    jacobian = x.new_zeros(*x.shape[:-1], 4, 6)
    jacobian[..., 0, :3] = x[..., :3] / radius.unsqueeze(-1)
    jacobian[..., 1, 3] = 1.0
    jacobian[..., 2, 4] = 1.0
    jacobian[..., 3, 5] = 1.0
    return jacobian


def covariance_projection(covariance, min_eigenvalue=1e-8):
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(min_eigenvalue)
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def normalized_structure(covariance):
    return covariance / torch.trace(covariance).clamp_min(1e-12)


def estimate_structure_scale(covariance, structure):
    """Frobenius projection argmin_a ||covariance - a*structure||_F."""
    numerator = torch.sum(covariance * structure)
    denominator = torch.sum(structure.square()).clamp_min(1e-12)
    return (numerator / denominator).clamp_min(1e-12)


def transition_builder(F_mat, B_mat, acc_batch):
    def f(x):
        x2 = x.squeeze(-1)
        return (x2 @ F_mat.T + acc_batch @ B_mat.T).unsqueeze(-1)

    return f


def knet_args(batch_size, device, in_mult, out_mult):
    return SimpleNamespace(
        use_cuda=device.type == "cuda",
        device=str(device),
        n_batch=batch_size,
        in_mult_KNet=in_mult,
        out_mult_KNet=out_mult,
        use_context_mod=True,
        knet_trainable=False,
    )


def build_adaptive_pair(
    base_state,
    batch_size,
    dt,
    device,
    in_mult,
    out_mult,
    hnet_hidden,
    prior_Q,
    prior_R,
    prior_P,
):
    dynamics = AcceleratedMovementModel(dt, device=device)
    state_dim, obs_dim, _, F_mat, _, B_mat = dynamics.get_dynamics()
    dummy_acc = torch.zeros(batch_size, B_mat.size(1), device=device)
    sys_model = KNetSysModel(
        transition_builder(F_mat, B_mat, dummy_acc),
        measurement_from_state,
        state_dim,
        obs_dim,
        prior_Q.squeeze(0),
        prior_P.squeeze(0),
        prior_R.squeeze(0),
    )
    mnet = KalmanNetNN().to(device)
    mnet.NNBuild(
        sys_model,
        knet_args(batch_size, device, in_mult, out_mult),
        frozen_weights=base_state,
    )
    gain_size = sum(value for key, value in mnet.cm_shape.items() if "gain" in key)
    shift_size = sum(value for key, value in mnet.cm_shape.items() if "shift" in key)
    if gain_size != shift_size:
        raise RuntimeError("Adaptive-KNet context gain/shift sizes differ.")
    # The reference Adaptive-KNet uses the scalar SoW = q^2 / r^2.
    hnet = HyperNetwork(1, gain_size, hidden_size=hnet_hidden).to(device)
    return hnet, mnet, F_mat, B_mat


def rollout(
    mnet,
    F_mat,
    B_mat,
    z,
    acc,
    x0,
    cm_shift,
    cm_gain,
    return_diagnostics=False,
):
    batch_size, seq_len, _ = z.shape
    mnet.batch_size = batch_size
    mnet.h = measurement_from_state
    mnet.InitSequence(x0, seq_len)
    mnet.init_hidden()
    estimates = []
    gains = []
    innovations = []
    for time_index in range(seq_len):
        mnet.f = transition_builder(F_mat, B_mat, acc_at(acc, time_index))
        estimate = mnet(
            z[:, time_index].unsqueeze(-1),
            weights_cm_gain=cm_gain,
            weights_cm_shift=cm_shift,
        )
        estimates.append(estimate.squeeze(-1))
        if return_diagnostics:
            gains.append(mnet.KGain.clone())
            innovations.append(mnet.dy.squeeze(-1).clone())
    estimates = torch.stack(estimates, dim=1)
    if not return_diagnostics:
        return estimates
    return estimates, {
        "kalman_gain": torch.stack(gains, dim=1),
        "innovation": torch.stack(innovations, dim=1),
    }


def sample_segment(condition, batch_size, segment_len, device):
    num_trajectories, total_len, _ = condition["true_states"].shape
    if batch_size > num_trajectories:
        raise ValueError("Adaptive-KNet batch size exceeds condition trajectories.")
    if segment_len > total_len:
        raise ValueError("segment_len exceeds trajectory length.")
    indices = torch.randperm(num_trajectories)[:batch_size]
    start = int(torch.randint(0, total_len - segment_len + 1, (1,)).item())
    end = start + segment_len
    acc = condition["acc_true"][indices]
    if acc.dim() == 3:
        acc = acc[:, start:end]
    x = condition["true_states"][indices, start:end].to(device)
    return (
        x,
        condition["measurements"][indices, start:end].to(device),
        acc.to(device),
        x[:, 0].unsqueeze(-1),
    )


def train_hypernetwork(
    hnet,
    mnet,
    F_mat,
    B_mat,
    conditions,
    args,
    device,
):
    optimizer = torch.optim.AdamW(
        hnet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        hnet.train()
        mnet.eval()
        epoch_loss = 0.0
        for _ in range(args.steps_per_epoch):
            condition = random.choice(conditions)
            x, z, acc, x0 = sample_segment(
                condition, args.batch_size_train, args.segment_len, device
            )
            hnet.init_hidden(device)
            cm_shift, cm_gain = hnet(condition["sow"].to(device))
            optimizer.zero_grad(set_to_none=True)
            estimates = rollout(
                mnet, F_mat, B_mat, z, acc, x0, cm_shift, cm_gain
            )
            loss = F.mse_loss(estimates, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hnet.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= max(1, args.steps_per_epoch)
        print(f"[Adaptive_KNet epoch {epoch}] train MSE={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "model_state_dict": hnet.state_dict(),
                    "epoch": epoch,
                    "train_mse": epoch_loss,
                    "base_checkpoint": str(args.base_checkpoint),
                    "sow_definition": "trace_ratio_q2_over_r2",
                    "sow_dimension": 1,
                },
                args.checkpoint,
            )


def innovation_moment_sums(estimates, measurements, diagnostics, r_old):
    """Reference-style innovation estimates accumulated over a batch."""
    batch_size, seq_len, _ = estimates.shape
    predicted_measurements = measurement_from_state(
        estimates.reshape(-1, estimates.size(-1)).unsqueeze(-1)
    ).squeeze(-1).reshape(batch_size, seq_len, -1)
    residual = measurements - predicted_measurements
    residual_outer = residual.unsqueeze(-1) @ residual.unsqueeze(-2)

    gains = diagnostics["kalman_gain"]
    jacobians = measurement_jacobian(estimates)
    hk = jacobians @ gains
    identity = torch.eye(
        measurements.size(-1), device=measurements.device, dtype=measurements.dtype
    ).view(1, 1, measurements.size(-1), measurements.size(-1))
    hph = torch.linalg.pinv(identity - hk) @ hk @ r_old
    r_sum = (residual_outer + hph).sum(dim=(0, 1))

    innovation = diagnostics["innovation"]
    innovation_outer_sum = (
        innovation.unsqueeze(-1) @ innovation.unsqueeze(-2)
    ).sum(dim=0)
    gain_sum = gains.sum(dim=0)
    return (
        gain_sum,
        innovation_outer_sum,
        r_sum,
        batch_size,
        batch_size * seq_len,
    )


@torch.no_grad()
def estimate_condition_sow(
    hnet,
    mnet,
    F_mat,
    B_mat,
    condition,
    prior_Q,
    prior_R,
    args,
    device,
):
    """Estimate scalar q^2/r^2 without reading the test dataset's Q or R."""
    q_old = covariance_projection(prior_Q)
    r_old = covariance_projection(prior_R)
    q_structure = normalized_structure(q_old)
    r_structure = normalized_structure(r_old)
    if args.initial_sow is None:
        sow = (torch.trace(q_old) / torch.trace(r_old)).reshape(1)
    else:
        sow = torch.tensor([args.initial_sow], device=device)

    min_sow = 10.0 ** (args.sow_min_db / 10.0)
    max_sow = 10.0 ** (args.sow_max_db / 10.0)
    sow = sow.clamp(min_sow, max_sow)
    num_trajectories = condition["measurements"].size(0)

    for iteration in range(args.sow_iterations):
        r_sum = torch.zeros_like(r_old)
        gain_sum = None
        innovation_outer_sum = None
        trajectory_count = 0
        residual_count = 0
        hnet.init_hidden(device)
        cm_shift, cm_gain = hnet(sow)
        for start in range(0, num_trajectories, args.batch_size_test):
            end = min(start + args.batch_size_test, num_trajectories)
            x = condition["true_states"][start:end].to(device)
            z = condition["measurements"][start:end].to(device)
            acc = condition["acc_true"][start:end].to(device)
            batch_size = end - start
            fixed_initial = torch.tensor(
                [0.0, 2.0, 3.0, 0.5, 0.5, 0.5], device=device
            )
            x0 = (
                x[:, 0].unsqueeze(-1)
                if args.true_x0
                else fixed_initial.unsqueeze(0)
                .expand(batch_size, -1)
                .unsqueeze(-1)
            )
            estimates, diagnostics = rollout(
                mnet,
                F_mat,
                B_mat,
                z,
                acc,
                x0,
                cm_shift,
                cm_gain,
                return_diagnostics=True,
            )
            (
                gain_batch,
                innovation_batch,
                r_batch,
                batch_trajectories,
                batch_residuals,
            ) = innovation_moment_sums(
                estimates, z, diagnostics, r_old
            )
            if gain_sum is None:
                gain_sum = torch.zeros_like(gain_batch)
                innovation_outer_sum = torch.zeros_like(innovation_batch)
            gain_sum += gain_batch
            innovation_outer_sum += innovation_batch
            r_sum += r_batch
            trajectory_count += batch_trajectories
            residual_count += batch_residuals

        mean_gain = gain_sum / max(trajectory_count, 1)
        innovation_moment = innovation_outer_sum / max(trajectory_count, 1)
        q_per_time = (
            mean_gain
            @ innovation_moment
            @ mean_gain.transpose(-1, -2)
        )
        q_est = covariance_projection(q_per_time.mean(dim=0))
        r_est = covariance_projection(r_sum / max(residual_count, 1))
        q_new = covariance_projection(
            args.sow_forget_factor * q_old
            + (1.0 - args.sow_forget_factor) * q_est
        )
        r_new = covariance_projection(
            args.sow_forget_factor * r_old
            + (1.0 - args.sow_forget_factor) * r_est
        )
        q2 = estimate_structure_scale(q_new, q_structure)
        r2 = estimate_structure_scale(r_new, r_structure)
        sow_new = (q2 / r2).clamp(min_sow, max_sow).reshape(1)
        print(
            f"[Adaptive_KNet class {condition['group_id']}] "
            f"SoW iteration {iteration + 1}: "
            f"q2={q2.item():.6g}, r2={r2.item():.6g}, "
            f"SoW={sow_new.item():.6g} "
            f"({10.0 * torch.log10(sow_new).item():.3f} dB)"
        )
        converged = torch.abs(sow_new - sow).item() <= args.sow_convergence
        sow, q_old, r_old = sow_new, q_new, r_new
        if converged:
            break
    return sow


@torch.no_grad()
def evaluate_dataset(
    hnet,
    mnet,
    F_mat,
    B_mat,
    raw_data,
    prior_Q,
    prior_R,
    args,
    device,
):
    hnet.eval()
    mnet.eval()
    flat = flatten_dataset(raw_data)
    estimates_all = torch.empty_like(flat["true_states"])
    offset = 0
    estimated_sow = {}
    for condition in grouped_conditions(raw_data, include_sow_label=False):
        num_trajectories = condition["true_states"].size(0)
        sow = estimate_condition_sow(
            hnet,
            mnet,
            F_mat,
            B_mat,
            condition,
            prior_Q,
            prior_R,
            args,
            device,
        )
        estimated_sow[condition["group_id"]] = sow.item()
        for start in range(0, num_trajectories, args.batch_size_test):
            end = min(start + args.batch_size_test, num_trajectories)
            x = condition["true_states"][start:end].to(device)
            z = condition["measurements"][start:end].to(device)
            acc = condition["acc_true"][start:end].to(device)
            batch_size = end - start
            fixed_initial = torch.tensor(
                [0.0, 2.0, 3.0, 0.5, 0.5, 0.5], device=device
            )
            x0 = (
                x[:, 0].unsqueeze(-1)
                if args.true_x0
                else fixed_initial.unsqueeze(0)
                .expand(batch_size, -1)
                .unsqueeze(-1)
            )
            hnet.init_hidden(device)
            cm_shift, cm_gain = hnet(sow)
            estimates = rollout(
                mnet, F_mat, B_mat, z, acc, x0, cm_shift, cm_gain
            )
            estimates_all[offset + start : offset + end] = estimates.cpu()
        offset += num_trajectories
    if offset != flat["true_states"].size(0):
        raise RuntimeError("Adaptive-KNet grouped evaluation lost trajectories.")
    result = summarize_estimates(estimates_all, flat["true_states"], flat)
    result["estimated_sow_by_class"] = estimated_sow
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=ROOT / "traj_dataset" / "maml_meta_train.pt",
    )
    parser.add_argument("--test-dataset", action="append", type=Path, default=None)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "results" / "kalmannet_current_best.pth",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "Adaptive_KNet" / "adaptive_knet_current_best.pth",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=ROOT / "results" / "Adaptive_KNet" / "adaptive_knet_current_results.pt",
    )
    parser.add_argument("--batch-size-train", type=int, default=16)
    parser.add_argument("--batch-size-test", type=int, default=16)
    parser.add_argument("--segment-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--in-mult", type=int, default=5)
    parser.add_argument("--out-mult", type=int, default=40)
    parser.add_argument("--hnet-hidden", type=int, default=256)
    parser.add_argument("--initial-sow", type=float, default=None)
    parser.add_argument("--sow-min-db", type=float, default=-20.0)
    parser.add_argument("--sow-max-db", type=float, default=10.0)
    parser.add_argument("--sow-iterations", type=int, default=1)
    parser.add_argument("--sow-forget-factor", type=float, default=0.3)
    parser.add_argument("--sow-convergence", type=float, default=1e-4)
    parser.add_argument("--true-x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sow_iterations < 1:
        raise ValueError("sow_iterations must be positive.")
    if not 0.0 <= args.sow_forget_factor <= 1.0:
        raise ValueError("sow_forget_factor must be in [0, 1].")
    if args.sow_min_db >= args.sow_max_db:
        raise ValueError("sow_min_db must be smaller than sow_max_db.")
    if args.initial_sow is not None and args.initial_sow <= 0.0:
        raise ValueError("initial_sow must be positive.")
    set_seed(args.seed)
    device = torch.device(args.device)
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(
            "Train KalmanNet first or pass --base-checkpoint to its checkpoint."
        )
    base_checkpoint = torch.load(args.base_checkpoint, map_location=device)
    base_state = base_checkpoint.get("model_state_dict", base_checkpoint)
    prior_Q = base_checkpoint.get("prior_Q", torch.eye(6).unsqueeze(0)).to(device)
    prior_R = base_checkpoint.get("prior_R", torch.eye(4).unsqueeze(0)).to(device)
    prior_P = base_checkpoint.get(
        "initial_covariance", torch.eye(6).mul(0.1).unsqueeze(0)
    ).to(device)
    hnet, mnet, F_mat, B_mat = build_adaptive_pair(
        base_state,
        args.batch_size_train,
        args.dt,
        device,
        args.in_mult,
        args.out_mult,
        args.hnet_hidden,
        prior_Q,
        prior_R,
        prior_P,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "both"):
        raw_train = load_raw_dataset(args.train_dataset)
        conditions = grouped_conditions(raw_train)
        train_hypernetwork(
            hnet, mnet, F_mat, B_mat, conditions, args, device
        )
    if args.mode in ("test", "both"):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        hnet.load_state_dict(checkpoint["model_state_dict"])
        test_paths = args.test_dataset or default_test_paths(args.train_dataset.parent)
        results = {}
        for path in test_paths:
            raw_test = load_raw_dataset(path)
            result = evaluate_dataset(
                hnet,
                mnet,
                F_mat,
                B_mat,
                raw_test,
                prior_Q.squeeze(0),
                prior_R.squeeze(0),
                args,
                device,
            )
            print_metrics(result, "Adaptive_KNet")
            results[result["dataset_type"]] = result
        config = vars(args).copy()
        for key in (
            "train_dataset",
            "base_checkpoint",
            "checkpoint",
            "result_output",
        ):
            config[key] = str(config[key])
        config["test_datasets"] = [str(path) for path in test_paths]
        save_results(args.result_output, "Adaptive_KNet", config, results)


if __name__ == "__main__":
    main()
