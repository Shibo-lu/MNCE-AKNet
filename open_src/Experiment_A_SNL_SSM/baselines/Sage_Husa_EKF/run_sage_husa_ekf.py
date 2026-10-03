# -*- coding: utf-8 -*-
"""Evaluate the Sage-Husa EKF on the current S0/S1/S3 datasets."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.current_dataset import (
    acc_at,
    default_test_paths,
    flatten_dataset,
    load_raw_dataset,
    print_metrics,
    save_results,
    summarize_estimates,
)
from sage_husa_ekf import SageHusaEKF


@torch.no_grad()
def run_filter(data, args, device):
    x_all = data["true_states"]
    z_all = data["measurements"]
    acc_all = data["acc_true"]
    num_trajectories, seq_len, state_dim = x_all.shape
    obs_dim = z_all.shape[-1]
    initial_covariance = torch.eye(state_dim, device=device) * args.p0
    process_noise = torch.eye(state_dim, device=device) * args.q0
    measurement_noise = torch.eye(obs_dim, device=device) * args.r0
    fixed_initial = torch.tensor(
        [0.0, 2.0, 3.0, 0.5, 0.5, 0.5], device=device
    )
    estimates_all = torch.empty_like(x_all)

    for start in range(0, num_trajectories, args.batch_size):
        end = min(start + args.batch_size, num_trajectories)
        x_seq = x_all[start:end].to(device)
        z_seq = z_all[start:end].to(device)
        acc = acc_all[start:end].to(device)
        batch_size = end - start
        x0 = (
            x_seq[:, 0]
            if args.true_x0
            else fixed_initial.unsqueeze(0).expand(batch_size, -1).clone()
        )
        ekf = SageHusaEKF(
            initial_state=x0,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=args.dt,
            b=args.b,
            schedule=args.schedule,
            warmup_steps=args.warmup_steps,
            diagonal_only=(not args.full_cov),
            adapt_q_mean=args.adapt_q_mean,
            adapt_r_mean=args.adapt_r_mean,
            adapt_q=(not args.fix_q),
            adapt_r=(not args.fix_r),
            q_bounds=(args.min_cov, args.max_cov),
            r_bounds=(args.min_cov, args.max_cov),
            device=device,
        )
        estimates = []
        for time_index in range(seq_len):
            ekf.step(
                z_seq[:, time_index], u=acc_at(acc, time_index)
            )
            estimates.append(ekf.get_state())
        estimates_all[start:end] = torch.stack(estimates, dim=1).cpu()
    return summarize_estimates(estimates_all, x_all, data)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "traj_dataset1")
    parser.add_argument(
        "--test-dataset", action="append", type=Path, default=None
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=ROOT / "results" / "sage_husa_current_results.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--schedule",
        default="running_average",
        choices=("running_average", "classic"),
    )
    parser.add_argument("--b", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--q0", type=float, default=1.0)
    parser.add_argument("--r0", type=float, default=1.0)
    parser.add_argument("--min-cov", type=float, default=1e-8)
    parser.add_argument("--max-cov", type=float, default=1e6)
    parser.add_argument(
        "--full-cov",
        action="store_true",
        help="Estimate full Q/R matrices. P and S always retain full covariance.",
    )
    parser.add_argument("--adapt-q-mean", action="store_true")
    parser.add_argument("--adapt-r-mean", action="store_true")
    parser.add_argument("--fix-q", action="store_true")
    parser.add_argument("--fix-r", action="store_true")
    parser.add_argument("--true-x0", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    device = torch.device(args.device)
    test_paths = args.test_dataset or default_test_paths(args.data_dir)
    results = {}
    for path in test_paths:
        data = flatten_dataset(load_raw_dataset(path))
        result = run_filter(data, args, device)
        print_metrics(result, "Sage_Husa_EKF")
        results[result["dataset_type"]] = result
    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["test_datasets"] = [str(path) for path in test_paths]
    config["result_output"] = str(args.result_output)
    save_results(args.result_output, "Sage_Husa_EKF", config, results)


if __name__ == "__main__":
    main()
