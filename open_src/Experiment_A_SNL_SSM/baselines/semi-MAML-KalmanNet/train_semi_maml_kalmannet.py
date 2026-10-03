# -*- coding: utf-8 -*-
"""Train and evaluate semi-MAML-KalmanNet on current MAML/S0/S1/S3 data."""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(__file__).resolve().parent
for path in (ROOT, LOCAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Dynamics import AcceleratedMovementModel
from baselines.current_dataset import (
    default_test_paths,
    load_raw_dataset,
    print_metrics,
    save_results,
    summarize_estimates,
)
from filter import KalmanNetFilter
from learner import Learner
from meta import (
    FIXED_INITIAL_STATE,
    adapted_query_loss,
    initial_states,
    meta_update,
)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tasks(raw_data):
    states = torch.as_tensor(raw_data["true_states"]).float().cpu()
    measurements = torch.as_tensor(raw_data["measurements"]).float().cpu()
    accelerations = torch.as_tensor(raw_data["acc_true"]).float().cpu()
    if states.dim() != 4:
        raise ValueError(
            "Meta-training data must have shape [task, trajectory, time, state]."
        )
    ratios = torch.as_tensor(
        raw_data.get("trace_ratio", torch.ones(states.size(0)))
    ).float().flatten()
    class_ids = torch.as_tensor(
        raw_data.get("class_id", torch.arange(states.size(0)))
    ).long().flatten()
    if ratios.numel() != states.size(0) or class_ids.numel() != states.size(0):
        raise ValueError("trace_ratio/class_id must contain one value per task.")
    return [
        {
            "task_id": task_index,
            "class_id": int(class_ids[task_index].item()),
            "trace_ratio": float(ratios[task_index].item()),
            "true_states": states[task_index],
            "measurements": measurements[task_index],
            "acc_true": accelerations[task_index],
        }
        for task_index in range(states.size(0))
    ]


def split_tasks_by_class(tasks, validation_tasks_per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    train_tasks = []
    validation_tasks = []
    class_ids = sorted({task["class_id"] for task in tasks})
    for class_id in class_ids:
        class_tasks = [task for task in tasks if task["class_id"] == class_id]
        if len(class_tasks) < 2:
            raise ValueError("Each class needs at least two meta-training tasks.")
        order = torch.randperm(len(class_tasks), generator=generator).tolist()
        validation_count = min(
            max(1, validation_tasks_per_class), len(class_tasks) - 1
        )
        validation_tasks.extend(class_tasks[index] for index in order[:validation_count])
        train_tasks.extend(class_tasks[index] for index in order[validation_count:])
    return train_tasks, validation_tasks


def sample_meta_batch(tasks, batch_size):
    if batch_size <= len(tasks):
        indices = torch.randperm(len(tasks))[:batch_size].tolist()
    else:
        indices = torch.randint(0, len(tasks), (batch_size,)).tolist()
    return [tasks[index] for index in indices]


def meta_config(args, sequence_length):
    segment_len = min(args.segment_len, sequence_length)
    return {
        "inner_lr": args.inner_lr,
        "standard_inner_lr": args.standard_inner_lr,
        "inner_steps": args.inner_steps,
        "inner_batch_size": args.inner_batch_size,
        "query_batch_size": args.query_batch_size,
        "support_size": args.support_size,
        "query_size": args.query_size,
        "segment_len": segment_len,
        "grad_clip": args.grad_clip,
        "true_x0": args.true_x0,
        "msg_step_weights": args.msg_step_weights,
    }


def train_meta_model(
    model,
    filter_runner,
    train_tasks,
    validation_tasks,
    args,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    config = meta_config(args, train_tasks[0]["true_states"].size(1))
    total_steps = args.epochs * args.steps_per_epoch
    msg_steps = int(total_steps * args.msg_fraction)
    global_step = 0
    best_validation = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        support_total = 0.0
        query_total = 0.0
        msg_count = 0
        for _ in range(args.steps_per_epoch):
            use_msg = global_step < msg_steps
            selected_tasks = sample_meta_batch(train_tasks, args.meta_batch_size)
            metrics = meta_update(
                model,
                filter_runner,
                optimizer,
                selected_tasks,
                config,
                device,
                use_msg,
            )
            support_total += metrics["support_mse"]
            query_total += metrics["query_mse"]
            msg_count += int(use_msg)
            global_step += 1

        support_mean = support_total / args.steps_per_epoch
        query_mean = query_total / args.steps_per_epoch
        record = {
            "epoch": epoch,
            "support_mse": support_mean,
            "query_mse": query_mean,
            "msg_steps": msg_count,
        }

        should_validate = epoch % args.validation_interval == 0 or epoch == args.epochs
        if should_validate:
            validation_values = [
                adapted_query_loss(
                    model, filter_runner, task, config, device
                )
                for task in validation_tasks
            ]
            validation_mse = float(np.mean(validation_values))
            record["validation_mse"] = validation_mse
            if validation_mse < best_validation:
                best_validation = validation_mse
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "validation_mse": validation_mse,
                        "state_dim": model.state_dim,
                        "measurement_dim": model.measurement_dim,
                        "nonlinear": model.nonlinear,
                        "config": serializable_config(args),
                    },
                    args.checkpoint,
                )
        history.append(record)
        validation_text = (
            f", val={record['validation_mse']:.6f}"
            if "validation_mse" in record
            else ""
        )
        phase = "MSG" if msg_count == args.steps_per_epoch else (
            "FOMAML" if msg_count == 0 else "MSG->FOMAML"
        )
        print(
            f"[semi-MAML epoch {epoch} {phase}] "
            f"support={support_mean:.6f}, query={query_mean:.6f}"
            f"{validation_text}"
        )
    return history


def fixed_initial(batch_size, device):
    return torch.tensor(FIXED_INITIAL_STATE, device=device).unsqueeze(0).expand(
        batch_size, -1
    )


def semi_supervised_adapt(
    base_model,
    filter_runner,
    measurements,
    accelerations,
    args,
    device,
):
    """Measurement-only adaptation used by semi-MAML-KalmanNet."""
    task_model = copy.deepcopy(base_model).to(device)
    task_model.train()
    optimizer = torch.optim.Adam(task_model.parameters(), lr=args.adapt_lr)
    sequence_length = measurements.size(1)
    adapt_length = (
        sequence_length
        if args.adapt_segment_len <= 0
        else min(sequence_length, args.adapt_segment_len)
    )
    if adapt_length < 3:
        raise ValueError("adapt_segment_len must provide at least three samples.")
    loss_history = []
    for _ in range(args.adapt_steps):
        if args.adapt_batch_size >= measurements.size(0):
            indices = torch.arange(measurements.size(0))
        else:
            indices = torch.randperm(measurements.size(0))[: args.adapt_batch_size]
        measurement_batch = measurements[indices, :adapt_length].to(device)
        acceleration_batch = accelerations[indices]
        if acceleration_batch.dim() == 3:
            acceleration_batch = acceleration_batch[:, :adapt_length]
        acceleration_batch = acceleration_batch.to(device)
        loss = filter_runner.measurement_prediction_loss(
            task_model,
            measurement_batch,
            acceleration_batch,
            fixed_initial(indices.numel(), device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model.parameters(), args.adapt_grad_clip)
        optimizer.step()
        loss_history.append(loss.detach().item())
    return task_model, loss_history


@torch.no_grad()
def evaluate_model(
    model,
    filter_runner,
    states,
    measurements,
    accelerations,
    batch_size,
    true_x0,
    device,
):
    model.eval()
    estimates = torch.empty_like(states)
    for start in range(0, states.size(0), batch_size):
        end = min(start + batch_size, states.size(0))
        state_batch = states[start:end].to(device)
        measurement_batch = measurements[start:end].to(device)
        acceleration_batch = accelerations[start:end].to(device)
        x0 = (
            initial_states(state_batch, True)
            if true_x0
            else fixed_initial(end - start, device)
        )
        output = filter_runner.rollout(
            model,
            measurement_batch,
            acceleration_batch,
            x0,
            query_hidden=True,
        )
        estimates[start:end] = output["states"].cpu()
    return estimates


def class_metadata(raw_data, class_index, count):
    ratio_values = raw_data.get("class_trace_ratio")
    frequency_values = raw_data.get("class_frequency")
    ratio = (
        float(torch.as_tensor(ratio_values)[class_index].item())
        if ratio_values is not None
        else float("nan")
    )
    frequency = (
        float(torch.as_tensor(frequency_values)[class_index].item())
        if frequency_values is not None
        else float("nan")
    )
    return (
        torch.full((count,), class_index, dtype=torch.long),
        torch.full((count,), ratio),
        torch.full((count,), frequency),
    )


def evaluate_dataset(base_model, filter_runner, raw_data, args, device):
    states = torch.as_tensor(raw_data["true_states"]).float().cpu()
    measurements = torch.as_tensor(raw_data["measurements"]).float().cpu()
    accelerations = torch.as_tensor(raw_data["acc_true"]).float().cpu()
    if states.dim() != 4:
        raise ValueError("Test data must have shape [class, trajectory, time, state].")

    pre_estimates = []
    post_estimates = []
    evaluation_states = []
    class_ids = []
    ratios = []
    frequencies = []
    adaptation = {}
    generator = torch.Generator().manual_seed(args.seed + 1000)

    for class_index in range(states.size(0)):
        trajectory_count = states.size(1)
        if args.adapt:
            if args.adapt_trajectories >= trajectory_count:
                raise ValueError(
                    "adapt_trajectories must leave at least one evaluation trajectory."
                )
            order = torch.randperm(trajectory_count, generator=generator)
            support_indices = order[: args.adapt_trajectories]
            evaluation_indices = order[args.adapt_trajectories :]
            adapted_model, loss_history = semi_supervised_adapt(
                base_model,
                filter_runner,
                measurements[class_index, support_indices],
                accelerations[class_index, support_indices],
                args,
                device,
            )
        else:
            support_indices = torch.empty(0, dtype=torch.long)
            evaluation_indices = torch.arange(trajectory_count)
            adapted_model = base_model
            loss_history = []

        class_states = states[class_index, evaluation_indices]
        class_measurements = measurements[class_index, evaluation_indices]
        class_accelerations = accelerations[class_index, evaluation_indices]
        pre_estimates.append(
            evaluate_model(
                base_model,
                filter_runner,
                class_states,
                class_measurements,
                class_accelerations,
                args.batch_size_test,
                args.true_x0,
                device,
            )
        )
        post_estimates.append(
            evaluate_model(
                adapted_model,
                filter_runner,
                class_states,
                class_measurements,
                class_accelerations,
                args.batch_size_test,
                args.true_x0,
                device,
            )
        )
        evaluation_states.append(class_states)
        metadata = class_metadata(raw_data, class_index, evaluation_indices.numel())
        class_ids.append(metadata[0])
        ratios.append(metadata[1])
        frequencies.append(metadata[2])
        adaptation[class_index] = {
            "support_indices": support_indices,
            "evaluation_indices": evaluation_indices,
            "measurement_prediction_loss": torch.tensor(loss_history),
        }

    metric_data = {
        "dataset_type": str(raw_data.get("dataset_type", "test")),
        "class_id": torch.cat(class_ids),
        "class_trace_ratio": torch.cat(ratios),
        "class_frequency": torch.cat(frequencies),
    }
    true_states = torch.cat(evaluation_states)
    pre_result = summarize_estimates(
        torch.cat(pre_estimates), true_states, metric_data
    )
    post_result = summarize_estimates(
        torch.cat(post_estimates), true_states, metric_data
    )
    post_result["pre_adaptation"] = pre_result
    post_result["adaptation"] = adaptation
    post_result["evaluation_protocol"] = "disjoint_support_and_query_trajectories"
    return post_result


def serializable_config(args):
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list) and value and isinstance(value[0], Path):
            config[key] = [str(path) for path in value]
    return config


def parse_step_weights(value):
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights or any(weight < 0 for weight in weights):
        raise argparse.ArgumentTypeError("MSG weights must be non-negative values.")
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test", "both"), default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "traj_dataset1")
    parser.add_argument("--train-dataset", type=Path, default=None)
    parser.add_argument("--test-dataset", action="append", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results" / "semi-MAML-KalmanNet" / "best.pth",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=ROOT / "results" / "semi-MAML-KalmanNet" / "test_results.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--support-size", type=int, default=16)
    parser.add_argument("--query-size", type=int, default=16)
    parser.add_argument("--inner-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--inner-lr", type=float, default=5e-4)
    parser.add_argument("--standard-inner-lr", type=float, default=5e-4)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--msg-fraction", type=float, default=0.5)
    parser.add_argument(
        "--msg-step-weights",
        type=parse_step_weights,
        default=parse_step_weights("0.3,0.3,0.2,0.1,0.1,0.01"),
    )
    parser.add_argument("--segment-len", type=int, default=64)
    parser.add_argument("--validation-tasks-per-class", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--adapt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-trajectories", type=int, default=25)
    parser.add_argument("--adapt-steps", type=int, default=16)
    parser.add_argument("--adapt-batch-size", type=int, default=4)
    parser.add_argument("--adapt-segment-len", type=int, default=50)
    parser.add_argument("--adapt-lr", type=float, default=9.6e-4)
    parser.add_argument("--adapt-grad-clip", type=float, default=10.0)
    parser.add_argument("--batch-size-test", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--true-x0", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args):
    positive_names = (
        "epochs",
        "steps_per_epoch",
        "meta_batch_size",
        "support_size",
        "query_size",
        "inner_batch_size",
        "query_batch_size",
        "inner_steps",
        "segment_len",
        "validation_interval",
        "adapt_steps",
        "adapt_batch_size",
        "batch_size_test",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if not 0.0 <= args.msg_fraction <= 1.0:
        raise ValueError("msg_fraction must be in [0, 1].")
    if args.adapt_trajectories < 1:
        raise ValueError("adapt_trajectories must be positive.")


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    train_path = args.train_dataset or args.data_dir / "maml_meta_train.pt"

    dynamics = AcceleratedMovementModel(args.dt, device=device)
    state_dim, measurement_dim, _, transition, _, control = dynamics.get_dynamics()
    filter_runner = KalmanNetFilter(transition, control, device)
    model = Learner(state_dim, measurement_dim, nonlinear=True).to(device)

    if args.mode in ("train", "both"):
        raw_train = load_raw_dataset(train_path)
        tasks = build_tasks(raw_train)
        train_tasks, validation_tasks = split_tasks_by_class(
            tasks, args.validation_tasks_per_class, args.seed
        )
        if args.support_size + args.query_size > tasks[0]["true_states"].size(0):
            raise ValueError("support_size + query_size exceeds trajectories per task.")
        train_meta_model(
            model,
            filter_runner,
            train_tasks,
            validation_tasks,
            args,
            device,
        )

    if args.mode in ("test", "both"):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_paths = args.test_dataset or default_test_paths(args.data_dir)
        results = {}
        for test_path in test_paths:
            raw_test = load_raw_dataset(test_path)
            result = evaluate_dataset(
                model, filter_runner, raw_test, args, device
            )
            print_metrics(result["pre_adaptation"], "MAML-init")
            print_metrics(result, "semi-MAML-KalmanNet")
            results[result["dataset_type"]] = result
        config = serializable_config(args)
        config["train_dataset"] = str(train_path)
        config["test_datasets"] = [str(path) for path in test_paths]
        save_results(
            args.result_output,
            "semi-MAML-KalmanNet",
            config,
            results,
        )


if __name__ == "__main__":
    main()
