# -*- coding: utf-8 -*-
"""Train and evaluate ARKFNet on the current MAML/S0/S1/S3 datasets."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.ARKFNet.arkfnet_model import LearnerARKFNet
from baselines.ARKFNet.filter import ProjectARKFNetFilter
from baselines.current_dataset import (
    acc_at,
    default_test_paths,
    flatten_dataset,
    load_raw_dataset,
    mean_covariance,
    print_metrics,
    save_results,
    summarize_estimates,
)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_corrupt_measurements(z, std, probability):
    if std <= 0.0 or probability <= 0.0:
        return z
    mask = (
        torch.rand(z.size(0), z.size(1), 1, device=z.device) < probability
    ).float()
    return z + torch.randn_like(z) * std * mask


def sample_batch(data, batch_size, segment_len, device):
    num_trajectories, total_len, _ = data["true_states"].shape
    if batch_size > num_trajectories:
        raise ValueError("Training batch size exceeds the number of trajectories.")
    if segment_len > total_len:
        raise ValueError("segment_len exceeds the trajectory length.")
    indices = torch.randperm(num_trajectories)[:batch_size]
    start = int(torch.randint(0, total_len - segment_len + 1, (1,)).item())
    end = start + segment_len
    acc = data["acc_true"][indices]
    if acc.dim() == 3:
        acc = acc[:, start:end]
    return (
        data["true_states"][indices, start:end].to(device),
        data["measurements"][indices, start:end].to(device),
        acc.to(device),
    )


def make_optimizers(model, learning_rate):
    first = (
        list(model.l1.parameters())
        + list(model.gru1.parameters())
        + list(model.l2.parameters())
    )
    second = (
        list(model.l3.parameters())
        + list(model.gru2.parameters())
        + list(model.l4.parameters())
    )
    return [
        torch.optim.Adam(first, lr=learning_rate),
        torch.optim.Adam(second, lr=learning_rate),
    ]


def train_model(model, filter_runner, data, args, fixed_initial, checkpoint_path):
    optimizers = make_optimizers(model, args.learning_rate)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step in range(args.steps_per_epoch):
            x_seg, z_seg, acc = sample_batch(
                data, args.batch_size_train, args.segment_len, model.device
                if hasattr(model, "device")
                else next(model.parameters()).device
            )
            z_seg = maybe_corrupt_measurements(
                z_seg, args.corrupt_std, args.corrupt_probability
            )
            optimizer = optimizers[step % len(optimizers)]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = filter_runner.train_loss(
                x_seg,
                z_seg,
                acc,
                model,
                use_true_x0=args.true_x0,
                fixed_initial_state=fixed_initial,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite ARKFNet loss at epoch={epoch}, step={step}."
                )
            loss.backward()
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= max(1, args.steps_per_epoch)
        print(f"[ARKFNet epoch {epoch}] train MSE={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_mse": epoch_loss,
                    "process_noise": filter_runner.Q.cpu(),
                    "measurement_noise": filter_runner.R.cpu(),
                    "initial_covariance": filter_runner.P0.cpu(),
                },
                checkpoint_path,
            )


@torch.no_grad()
def evaluate(model, filter_runner, data, args, fixed_initial):
    model.eval()
    x_all = data["true_states"]
    z_all = data["measurements"]
    acc_all = data["acc_true"]
    estimates_all = torch.empty_like(x_all)
    device = next(model.parameters()).device
    for start in range(0, x_all.size(0), args.batch_size_test):
        end = min(start + args.batch_size_test, x_all.size(0))
        x_seq = x_all[start:end].to(device)
        z_seq = z_all[start:end].to(device)
        acc = acc_all[start:end].to(device)
        batch_size = end - start
        initial = (
            x_seq[:, 0]
            if args.true_x0
            else fixed_initial.unsqueeze(0).expand(batch_size, -1).clone()
        )
        filter_runner.reset(batch_size, initial)
        model.reset(batch_size, device)
        estimates = []
        for time_index in range(x_seq.size(1)):
            estimates.append(
                filter_runner.step_arkfnet(
                    z_seq[:, time_index].unsqueeze(-1),
                    acc_at(acc, time_index),
                    model,
                )
            )
        estimates_all[start:end] = torch.stack(estimates, dim=1).cpu()
    return summarize_estimates(estimates_all, x_all, data)


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
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "ARKFNet" / "arkfnet_current_best.pth",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=ROOT / "results" / "ARKFNet" / "arkfnet_current_results.pt",
    )
    parser.add_argument("--batch-size-train", type=int, default=10)
    parser.add_argument("--batch-size-test", type=int, default=16)
    parser.add_argument("--segment-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--slide-window", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--q0", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--use-data-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--corrupt-std", type=float, default=0.0)
    parser.add_argument("--corrupt-probability", type=float, default=0.0)
    parser.add_argument("--true-x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fixed_initial = torch.tensor(
        [0.0, 2.0, 3.0, 0.5, 0.5, 0.5], device=device
    )
    raw_train = None
    if args.mode in ("train", "both"):
        raw_train = load_raw_dataset(args.train_dataset)
        train_data = flatten_dataset(raw_train)
        process_noise = (
            mean_covariance(raw_train, "Q", 6)
            if args.use_data_priors
            else torch.eye(6) * args.q0
        )
        measurement_noise = (
            mean_covariance(raw_train, "R", 4)
            if args.use_data_priors
            else torch.eye(4) * args.r0
        )
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        process_noise = checkpoint.get("process_noise", torch.eye(6) * args.q0)
        measurement_noise = checkpoint.get(
            "measurement_noise", torch.eye(4) * args.r0
        )
    initial_covariance = (
        torch.eye(6) * args.p0
        if raw_train is not None
        else checkpoint.get("initial_covariance", torch.eye(6) * args.p0)
    )
    model = LearnerARKFNet(
        6,
        4,
        hidden_scale=args.hidden_scale,
        seq_window=args.slide_window,
        bidirectional=True,
    ).to(device)
    filter_runner = ProjectARKFNetFilter(
        dt=args.dt,
        process_noise=process_noise.to(device),
        measurement_noise=measurement_noise.to(device),
        initial_covariance=initial_covariance.to(device),
        slide_window=args.slide_window,
        device=device,
    )
    if args.mode in ("train", "both"):
        train_model(
            model,
            filter_runner,
            train_data,
            args,
            fixed_initial,
            args.checkpoint,
        )
    if args.mode in ("test", "both"):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_paths = args.test_dataset or default_test_paths(args.train_dataset.parent)
        results = {}
        for path in test_paths:
            data = flatten_dataset(load_raw_dataset(path))
            result = evaluate(model, filter_runner, data, args, fixed_initial)
            print_metrics(result, "ARKFNet")
            results[result["dataset_type"]] = result
        config = vars(args).copy()
        config["train_dataset"] = str(args.train_dataset)
        config["test_datasets"] = [str(path) for path in test_paths]
        config["checkpoint"] = str(args.checkpoint)
        config["result_output"] = str(args.result_output)
        save_results(args.result_output, "ARKFNet", config, results)


if __name__ == "__main__":
    main()
