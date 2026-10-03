# -*- coding: utf-8 -*-
"""Evaluate adaptive EKF or fixed-covariance EKF on S0/S1/S3 datasets."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_ekf import WindowAdaptiveEKF
from baselines.current_dataset import (
    acc_at,
    default_test_paths,
    flatten_dataset,
    load_raw_dataset,
    print_metrics,
    save_results,
    summarize_estimates,
)


def batch_initial(initial_state, batch_size, device):
    return initial_state.unsqueeze(0).expand(batch_size, -1).clone().to(device)


@torch.no_grad()
def run_filter(data, args, device):
    x_all = data["true_states"]
    z_all = data["measurements"]
    acc_all = data["acc_true"]
    num_trajectories, seq_len, state_dim = x_all.shape
    obs_dim = z_all.shape[-1]
    initial_covariance = torch.eye(state_dim, device=device) * args.p0
    use_adaptation = args.filter_type == "adaptive"
    if use_adaptation:
        process_noise = torch.eye(state_dim, device=device) * args.q0
        measurement_noise = torch.eye(obs_dim, device=device) * args.r0
    else:
        # Ordinary EKF keeps Q=I_6 and R=I_4 fixed for the whole trajectory.
        process_noise = torch.eye(state_dim, device=device)
        measurement_noise = torch.eye(obs_dim, device=device)
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
        x0 = x_seq[:, 0] if args.true_x0 else batch_initial(
            fixed_initial, batch_size, device
        )
        kwargs = dict(
            initial_state=x0,
            initial_covariance=initial_covariance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            dt=args.dt,
            window_size=args.window_size,
            min_window=args.min_window,
            q_bounds=(args.min_cov, args.max_cov),
            r_bounds=(args.min_cov, args.max_cov),
            adapt_noise=use_adaptation,
            device=device,
        )
        if args.forgetting_factor is not None:
            kwargs["forgetting_factor"] = args.forgetting_factor
        ekf = WindowAdaptiveEKF(**kwargs)
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
    parser.add_argument(
        "--filter-type",
        choices=("adaptive", "ekf"),
        default="adaptive",
        help="Select adaptive EKF or ordinary EKF with fixed Q=I and R=I.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "traj_dataset")
    parser.add_argument(
        "--test-dataset", action="append", type=Path, default=None
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=None,
        help="Output path. If omitted, a distinct filename is selected for each filter type.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--min-window", type=int, default=5)
    parser.add_argument("--forgetting-factor", type=float, default=None)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument(
        "--q0", type=float, default=1.0,
        help="Initial Q scale for adaptive mode; ignored in ordinary EKF mode.",
    )
    parser.add_argument(
        "--r0", type=float, default=1.0,
        help="Initial R scale for adaptive mode; ignored in ordinary EKF mode.",
    )
    parser.add_argument("--min-cov", type=float, default=1e-8)
    parser.add_argument("--max-cov", type=float, default=1e6)
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
    method_name = (
        "Adaptive_EKF" if args.filter_type == "adaptive" else "EKF_Identity_QR"
    )
    if args.result_output is None:
        result_filename = (
            "adaptive_ekf_current_results.pt"
            if args.filter_type == "adaptive"
            else "ekf_identity_qr_current_results.pt"
        )
        args.result_output = ROOT / "results" / result_filename
    device = torch.device(args.device)
    test_paths = args.test_dataset or default_test_paths(args.data_dir)
    results = {}
    for path in test_paths:
        data = flatten_dataset(load_raw_dataset(path))
        result = run_filter(data, args, device)
        print_metrics(result, method_name)
        results[result["dataset_type"]] = result
    config = vars(args).copy()
    config["data_dir"] = str(args.data_dir)
    config["test_datasets"] = [str(path) for path in test_paths]
    config["result_output"] = str(args.result_output)
    config["effective_Q"] = "adaptive" if args.filter_type == "adaptive" else "I_6"
    config["effective_R"] = "adaptive" if args.filter_type == "adaptive" else "I_4"
    save_results(args.result_output, method_name, config, results)


if __name__ == "__main__":
    main()
