# -*- coding: utf-8 -*-
"""Train and evaluate KalmanNet on the current MAML/S0/S1/S3 datasets."""

import argparse
import math
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

from baselines.KalmanNet.KalmanNet_nn import KalmanNetNN
from Dynamics import AcceleratedMovementModel
from device import get_device


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class KNetSysModel:
    def __init__(self, f, h, m, n, prior_Q, prior_Sigma, prior_S):
        self.f = f
        self.h = h
        self.m = m
        self.n = n
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S


def measurement_from_state(x):
    x2 = x.squeeze(-1)
    px, py, pz = x2[:, 0], x2[:, 1], x2[:, 2]
    radius = torch.sqrt(px.square() + py.square() + pz.square() + 1e-8)
    return torch.stack(
        [radius, x2[:, 3], x2[:, 4], x2[:, 5]], dim=-1
    ).unsqueeze(-1)


def transition_builder(F_mat, B_mat, acc_batch):
    def f(x):
        x2 = x.squeeze(-1)
        return (x2 @ F_mat.T + acc_batch @ B_mat.T).unsqueeze(-1)

    return f


def _acc_at(acc, time_index):
    return acc[:, time_index, :] if acc.dim() == 3 else acc


def build_kalmannet(
    batch_size,
    dt,
    initial_covariance,
    process_noise,
    measurement_noise,
    device,
):
    dynamics_model = AcceleratedMovementModel(dt, device=device)
    state_dim, obs_dim, _, F_mat, _, B_mat = dynamics_model.get_dynamics()
    dummy_acc = torch.zeros(batch_size, B_mat.shape[1], device=device)
    sys_model = KNetSysModel(
        f=transition_builder(F_mat, B_mat, dummy_acc),
        h=measurement_from_state,
        m=state_dim,
        n=obs_dim,
        prior_Q=process_noise.squeeze(0),
        prior_Sigma=initial_covariance.squeeze(0),
        prior_S=measurement_noise.squeeze(0),
    )
    args = SimpleNamespace(
        use_cuda=(device.type == "cuda"),
        device=str(device),
        n_batch=batch_size,
        in_mult_KNet=5,
        out_mult_KNet=40,
    )
    model = KalmanNetNN().to(device)
    model.NNBuild(sys_model, args)
    return model, F_mat, B_mat


def rollout_kalmannet(model, F_mat, B_mat, z_seq, acc, initial_state):
    batch_size, seq_len, _ = z_seq.shape
    model.batch_size = batch_size
    model.h = measurement_from_state
    x0 = initial_state.repeat(batch_size, 1).unsqueeze(-1)
    model.InitSequence(x0, seq_len)
    model.init_hidden_KNet()

    estimates = []
    for time_index in range(seq_len):
        model.f = transition_builder(F_mat, B_mat, _acc_at(acc, time_index))
        y_t = z_seq[:, time_index, :].unsqueeze(-1)
        estimates.append(model(y_t).squeeze(-1))
    return torch.stack(estimates, dim=1)


def _expand_class_metadata(data, task_shaped, num_groups, trajectories_per_group):
    total = num_groups * trajectories_per_group if task_shaped else num_groups
    if task_shaped:
        if "trajectory_class_id" in data:
            class_ids = data["trajectory_class_id"].flatten()
        elif "class_id" in data and data["class_id"].numel() == num_groups:
            class_ids = data["class_id"].repeat_interleave(trajectories_per_group)
        else:
            class_ids = torch.arange(num_groups).repeat_interleave(
                trajectories_per_group
            )
    elif "trajectory_class_id" in data:
        class_ids = data["trajectory_class_id"].flatten()
    elif "class_id" in data and data["class_id"].numel() == total:
        class_ids = data["class_id"].flatten()
    else:
        class_ids = torch.zeros(total, dtype=torch.long)

    def expand_value(primary_key, fallback_key=None):
        value = data.get(primary_key)
        if value is None and fallback_key is not None:
            value = data.get(fallback_key)
        if value is None:
            return torch.full((total,), float("nan"))
        value = torch.as_tensor(value).flatten()
        if value.numel() == total:
            return value.float()
        if task_shaped and value.numel() == num_groups:
            return value.repeat_interleave(trajectories_per_group).float()
        return torch.full((total,), float("nan"))

    return (
        class_ids.long(),
        expand_value("class_trace_ratio", "trace_ratio"),
        expand_value("class_frequency"),
    )


def flatten_current_dataset(data):
    """Convert [group, trajectory, time, ...] to trajectory-major tensors."""
    task_shaped = data["true_states"].dim() == 4
    if task_shaped:
        num_groups, trajectories_per_group = data["true_states"].shape[:2]
    else:
        num_groups = data["true_states"].size(0)
        trajectories_per_group = 1

    result = {}
    for key in ("true_states", "measurements", "acc_true"):
        value = data[key].detach().cpu()
        result[key] = value.flatten(0, 1) if task_shaped else value
    class_ids, ratios, frequencies = _expand_class_metadata(
        data, task_shaped, num_groups, trajectories_per_group
    )
    result["class_id"] = class_ids.cpu()
    result["class_trace_ratio"] = ratios.cpu()
    result["class_frequency"] = frequencies.cpu()
    result["dataset_type"] = str(data.get("dataset_type", "test"))
    return result


def mean_prior_covariance(data, key, dimension):
    value = data.get(key)
    if value is None:
        return torch.eye(dimension).unsqueeze(0)
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError(
            f"{key} must end in [{dimension}, {dimension}], got {tuple(value.shape)}."
        )
    return value.reshape(-1, dimension, dimension).mean(dim=0, keepdim=True)


def split_train_validation(data, validation_fraction, max_validation, seed):
    num_trajectories = data["true_states"].size(0)
    if num_trajectories < 2:
        raise ValueError("Training requires at least two trajectories.")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_trajectories, generator=generator)
    val_count = max(1, int(round(num_trajectories * validation_fraction)))
    val_count = min(val_count, max_validation, num_trajectories - 1)
    val_indices = permutation[:val_count]
    train_indices = permutation[val_count:]

    def subset(indices):
        return {
            key: value[indices]
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
            and value.size(0) == num_trajectories
            and key in ("true_states", "measurements", "acc_true")
        }

    return subset(train_indices), subset(val_indices)


def train_one_epoch_knet(
    model,
    F_mat,
    B_mat,
    optimizer,
    data,
    batch_size,
    segment_len,
    num_steps,
    grad_clip,
    device,
):
    model.train()
    true_states = data["true_states"]
    measurements = data["measurements"]
    acc_true = data["acc_true"]
    num_trajectories, total_len, _ = true_states.shape
    if total_len < segment_len:
        raise ValueError(
            f"segment_len={segment_len} exceeds sequence length {total_len}."
        )
    if batch_size > num_trajectories:
        raise ValueError(
            f"batch_size={batch_size} exceeds {num_trajectories} train trajectories."
        )

    total_loss = 0.0
    for _ in range(num_steps):
        indices = torch.randperm(num_trajectories)[:batch_size]
        x_seq = true_states[indices].to(device)
        z_seq = measurements[indices].to(device)
        acc = acc_true[indices].to(device)
        start = int(torch.randint(0, total_len - segment_len + 1, (1,)).item())
        end = start + segment_len
        x_target = x_seq[:, start:end]
        z_segment = z_seq[:, start:end]
        acc_segment = acc[:, start:end] if acc.dim() == 3 else acc

        model.batch_size = batch_size
        model.h = measurement_from_state
        model.InitSequence(x_seq[:, start].unsqueeze(-1), segment_len)
        model.init_hidden_KNet()
        optimizer.zero_grad(set_to_none=True)

        estimates = []
        for local_time in range(segment_len):
            model.f = transition_builder(
                F_mat, B_mat, _acc_at(acc_segment, local_time)
            )
            estimates.append(
                model(z_segment[:, local_time].unsqueeze(-1)).squeeze(-1)
            )
        estimates = torch.stack(estimates, dim=1)
        loss = F.mse_loss(estimates, x_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / max(1, num_steps)


@torch.no_grad()
def evaluate_knet(
    model,
    F_mat,
    B_mat,
    data,
    initial_state,
    batch_size,
    device,
    collect_states=True,
):
    model.eval()
    true_states = data["true_states"]
    measurements = data["measurements"]
    acc_true = data["acc_true"]
    num_trajectories = true_states.size(0)
    mse_per_trajectory = torch.empty(num_trajectories)
    pos_mse_per_trajectory = torch.empty(num_trajectories)
    vel_mse_per_trajectory = torch.empty(num_trajectories)
    estimated_states = torch.empty_like(true_states) if collect_states else None

    for start in range(0, num_trajectories, batch_size):
        end = min(start + batch_size, num_trajectories)
        x_seq = true_states[start:end].to(device)
        z_seq = measurements[start:end].to(device)
        acc = acc_true[start:end].to(device)
        estimates = rollout_kalmannet(
            model, F_mat, B_mat, z_seq, acc, initial_state
        )
        error = (estimates - x_seq).square()
        mse_per_trajectory[start:end] = error.mean(dim=(1, 2)).cpu()
        pos_mse_per_trajectory[start:end] = error[:, :, :3].mean(dim=(1, 2)).cpu()
        vel_mse_per_trajectory[start:end] = error[:, :, 3:].mean(dim=(1, 2)).cpu()
        if collect_states:
            estimated_states[start:end] = estimates.cpu()

    class_ids = data.get(
        "class_id", torch.zeros(num_trajectories, dtype=torch.long)
    )
    ratios = data.get(
        "class_trace_ratio", torch.full((num_trajectories,), float("nan"))
    )
    frequencies = data.get(
        "class_frequency", torch.full((num_trajectories,), float("nan"))
    )
    unique_classes = torch.unique(class_ids, sorted=True)
    class_metrics = {}
    class_mse = []
    class_pos_mse = []
    class_vel_mse = []
    class_ratios = []
    class_frequencies = []
    for class_tensor in unique_classes:
        class_id = int(class_tensor.item())
        indices = torch.where(class_ids == class_tensor)[0]
        finite_ratios = ratios[indices][torch.isfinite(ratios[indices])]
        finite_frequencies = frequencies[indices][
            torch.isfinite(frequencies[indices])
        ]
        ratio = finite_ratios.mean().item() if finite_ratios.numel() else float("nan")
        frequency = (
            finite_frequencies.mean().item()
            if finite_frequencies.numel()
            else float("nan")
        )
        mse = mse_per_trajectory[indices].mean().item()
        pos_mse = pos_mse_per_trajectory[indices].mean().item()
        vel_mse = vel_mse_per_trajectory[indices].mean().item()
        class_mse.append(mse)
        class_pos_mse.append(pos_mse)
        class_vel_mse.append(vel_mse)
        class_ratios.append(ratio)
        class_frequencies.append(frequency)
        class_metrics[class_id] = {
            "trace_ratio": ratio,
            "frequency_cycles_per_trajectory": frequency,
            "num_trajectories": int(indices.numel()),
            "mse": mse,
            "position_mse": pos_mse,
            "velocity_mse": vel_mse,
        }

    result = {
        "dataset_type": data.get("dataset_type", "test"),
        "mse_per_trajectory": mse_per_trajectory,
        "position_mse_per_trajectory": pos_mse_per_trajectory,
        "velocity_mse_per_trajectory": vel_mse_per_trajectory,
        "class_id_per_trajectory": class_ids,
        "trace_ratio_per_trajectory": ratios,
        "frequency_per_trajectory": frequencies,
        "mean_mse": mse_per_trajectory.mean().item(),
        "mean_position_mse": pos_mse_per_trajectory.mean().item(),
        "mean_velocity_mse": vel_mse_per_trajectory.mean().item(),
        "class_ids": unique_classes,
        "class_trace_ratios": torch.tensor(class_ratios),
        "class_frequencies": torch.tensor(class_frequencies),
        "class_mse": torch.tensor(class_mse),
        "class_position_mse": torch.tensor(class_pos_mse),
        "class_velocity_mse": torch.tensor(class_vel_mse),
        "class_metrics": class_metrics,
    }
    if collect_states:
        result["states"] = estimated_states
    return result


def print_test_metrics(result):
    dataset_name = result["dataset_type"]
    for class_id, metrics in result["class_metrics"].items():
        print(
            f"[{dataset_name} class {class_id}] "
            f"ratio={metrics['trace_ratio']:.6g}, "
            f"frequency={metrics['frequency_cycles_per_trajectory']:.6g}, "
            f"MSE={metrics['mse']:.6f}, "
            f"position={metrics['position_mse']:.6f}, "
            f"velocity={metrics['velocity_mse']:.6f}"
        )
    print(
        f"[{dataset_name} overall] MSE={result['mean_mse']:.6f}, "
        f"position={result['mean_position_mse']:.6f}, "
        f"velocity={result['mean_velocity_mse']:.6f}"
    )


def save_checkpoint(path, model, optimizer, epoch, validation_mse, config, priors):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_mse": validation_mse,
            "config": config,
            "prior_Q": priors["Q"].cpu(),
            "prior_R": priors["R"].cpu(),
            "initial_covariance": priors["P0"].cpu(),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=ROOT / "traj_dataset1" / "maml_meta_train.pt",
    )
    parser.add_argument(
        "--test-dataset",
        dest="test_datasets",
        action="append",
        type=Path,
        default=None,
        help="Repeat to test multiple datasets; defaults to S0, S1 and S3.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "kalmannet_current_best.pth",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=ROOT / "results" / "kalmannet_current_test_results.pt",
    )
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--segment-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-validation-trajectories", type=int, default=320)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device, _ = get_device()
    initial_state = torch.tensor(
        [[0.0, 2.0, 3.0, 0.5, 0.5, 0.5]],
        dtype=torch.float32,
        device=device,
    )

    default_tests = [
        ROOT / "traj_dataset1" / "maml_test_S0_fixed.pt",
        ROOT / "traj_dataset1" / "maml_test_S1_abrupt.pt",
        ROOT / "traj_dataset1" / "maml_test_S3_smooth.pt",
    ]
    test_paths = args.test_datasets or default_tests
    config = vars(args).copy()
    config["train_dataset"] = str(args.train_dataset)
    config["test_datasets"] = [str(path) for path in test_paths]
    config["checkpoint"] = str(args.checkpoint)
    config["result_output"] = str(args.result_output)

    model = None
    F_mat = None
    B_mat = None
    priors = None

    if args.mode in ("train", "both"):
        raw_train = torch.load(args.train_dataset, map_location="cpu")
        priors = {
            "Q": mean_prior_covariance(raw_train, "Q", 6),
            "R": mean_prior_covariance(raw_train, "R", 4),
            "P0": torch.eye(6).mul(0.1).unsqueeze(0),
        }
        flat_train = flatten_current_dataset(raw_train)
        train_data, validation_data = split_train_validation(
            flat_train,
            args.validation_fraction,
            args.max_validation_trajectories,
            args.seed,
        )
        print(
            f"train trajectories={train_data['true_states'].size(0)}, "
            f"validation trajectories={validation_data['true_states'].size(0)}"
        )
        model, F_mat, B_mat = build_kalmannet(
            args.batch_size_train,
            args.dt,
            priors["P0"].to(device),
            priors["Q"].to(device),
            priors["R"].to(device),
            device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_validation = math.inf
        bad_epochs = 0
        for epoch in range(args.epochs):
            train_mse = train_one_epoch_knet(
                model,
                F_mat,
                B_mat,
                optimizer,
                train_data,
                args.batch_size_train,
                args.segment_len,
                args.steps_per_epoch,
                args.grad_clip,
                device,
            )
            validation = evaluate_knet(
                model,
                F_mat,
                B_mat,
                validation_data,
                initial_state,
                args.batch_size_test,
                device,
                collect_states=False,
            )
            validation_mse = validation["mean_mse"]
            print(
                f"[epoch {epoch + 1}] train MSE={train_mse:.6f}, "
                f"validation MSE={validation_mse:.6f}"
            )
            if validation_mse < best_validation:
                best_validation = validation_mse
                bad_epochs = 0
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    epoch,
                    validation_mse,
                    config,
                    priors,
                )
            else:
                bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if args.mode in ("test", "both"):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        priors = {
            "Q": checkpoint.get("prior_Q", torch.eye(6).unsqueeze(0)).to(device),
            "R": checkpoint.get("prior_R", torch.eye(4).unsqueeze(0)).to(device),
            "P0": checkpoint.get(
                "initial_covariance", torch.eye(6).mul(0.1).unsqueeze(0)
            ).to(device),
        }
        model, F_mat, B_mat = build_kalmannet(
            args.batch_size_test,
            args.dt,
            priors["P0"],
            priors["Q"],
            priors["R"],
            device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        all_results = {}
        for test_path in test_paths:
            raw_test = torch.load(test_path, map_location="cpu")
            test_data = flatten_current_dataset(raw_test)
            result = evaluate_knet(
                model,
                F_mat,
                B_mat,
                test_data,
                initial_state,
                args.batch_size_test,
                device,
                collect_states=args.save_predictions,
            )
            print_test_metrics(result)
            name = result["dataset_type"]
            if name in all_results:
                name = test_path.stem
            all_results[name] = result

        args.result_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint": str(args.checkpoint),
                "config": config,
                "datasets": all_results,
            },
            args.result_output,
        )
        print(f"saved KalmanNet test results to {args.result_output}")


if __name__ == "__main__":
    main()
