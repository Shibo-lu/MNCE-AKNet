"""Shared loaders and metrics for the current task-structured datasets."""

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILENAMES = (
    "maml_test_S0_fixed.pt",
    "maml_test_S1_abrupt.pt",
    "maml_test_S3_smooth.pt",
)


def default_test_paths(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "traj_dataset"
    return [data_dir / filename for filename in DEFAULT_TEST_FILENAMES]


def load_raw_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return torch.load(path, map_location="cpu")


def _expand_metadata(data, task_shaped, num_groups, trajectories_per_group):
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

    def value_for(primary, fallback=None):
        value = data.get(primary)
        if value is None and fallback is not None:
            value = data.get(fallback)
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
        value_for("class_trace_ratio", "trace_ratio"),
        value_for("class_frequency"),
    )


def flatten_dataset(data):
    """Return CPU tensors with leading dimensions flattened to trajectory."""
    task_shaped = data["true_states"].dim() == 4
    if task_shaped:
        num_groups, trajectories_per_group = data["true_states"].shape[:2]
    else:
        num_groups = data["true_states"].size(0)
        trajectories_per_group = 1

    result = {}
    for key in ("true_states", "measurements", "acc_true"):
        value = torch.as_tensor(data[key]).float().cpu()
        result[key] = value.flatten(0, 1) if task_shaped else value
    class_ids, ratios, frequencies = _expand_metadata(
        data, task_shaped, num_groups, trajectories_per_group
    )
    result["class_id"] = class_ids.cpu()
    result["class_trace_ratio"] = ratios.cpu()
    result["class_frequency"] = frequencies.cpu()
    result["dataset_type"] = str(data.get("dataset_type", "test"))
    return result


def mean_covariance(data, key, dimension, default_scale=1.0):
    value = data.get(key)
    if value is None:
        return torch.eye(dimension) * default_scale
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.shape[-2:] != (dimension, dimension):
        raise ValueError(
            f"{key} must end in [{dimension}, {dimension}], got {tuple(value.shape)}."
        )
    return value.reshape(-1, dimension, dimension).mean(dim=0)


def _group_ratio_label(data, group_index, num_groups):
    """Read a simulation-condition label without inspecting Q/R tensors."""
    for key in ("trace_ratio", "class_trace_ratio"):
        value = data.get(key)
        if value is None:
            continue
        value = torch.as_tensor(value, dtype=torch.float32).flatten()
        if value.numel() == num_groups:
            return value[group_index].reshape(1)
    raise KeyError(
        "Adaptive-KNet training requires a per-task trace_ratio label; "
        "it is not inferred from the ground-truth Q/R tensors."
    )


def grouped_conditions(data, include_sow_label=True):
    """Return one condition per task/class without exposing test Q/R tensors."""
    if data["true_states"].dim() == 4:
        num_groups = data["true_states"].size(0)
        conditions = []
        for group_index in range(num_groups):
            condition = {
                "group_id": group_index,
                "true_states": data["true_states"][group_index].float().cpu(),
                "measurements": data["measurements"][group_index].float().cpu(),
                "acc_true": data["acc_true"][group_index].float().cpu(),
            }
            if include_sow_label:
                condition["sow"] = _group_ratio_label(
                    data, group_index, num_groups
                )
            conditions.append(condition)
        return conditions

    flat = flatten_dataset(data)
    conditions = []
    for class_tensor in torch.unique(flat["class_id"], sorted=True):
        indices = torch.where(flat["class_id"] == class_tensor)[0]
        condition = {
            "group_id": int(class_tensor.item()),
            "true_states": flat["true_states"][indices],
            "measurements": flat["measurements"][indices],
            "acc_true": flat["acc_true"][indices],
        }
        if include_sow_label:
            ratios = flat["class_trace_ratio"][indices]
            ratios = ratios[torch.isfinite(ratios)]
            if not ratios.numel():
                raise KeyError("No trace-ratio label is available for training.")
            condition["sow"] = ratios.mean().reshape(1)
        conditions.append(condition)
    return conditions


def acc_at(acc, time_index):
    return acc[:, time_index, :] if acc.dim() == 3 else acc


def summarize_estimates(estimates, true_states, data):
    estimates = estimates.detach().cpu()
    true_states = true_states.detach().cpu()
    error = (estimates - true_states).square()
    mse_per_trajectory = error.mean(dim=(1, 2))
    pos_mse_per_trajectory = error[:, :, :3].mean(dim=(1, 2))
    vel_mse_per_trajectory = error[:, :, 3:].mean(dim=(1, 2))
    class_ids = data.get(
        "class_id", torch.zeros(true_states.size(0), dtype=torch.long)
    ).cpu()
    ratios = data.get(
        "class_trace_ratio",
        torch.full((true_states.size(0),), float("nan")),
    ).cpu()
    frequencies = data.get(
        "class_frequency",
        torch.full((true_states.size(0),), float("nan")),
    ).cpu()
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
        finite_ratio = ratios[indices][torch.isfinite(ratios[indices])]
        finite_frequency = frequencies[indices][
            torch.isfinite(frequencies[indices])
        ]
        ratio = finite_ratio.mean().item() if finite_ratio.numel() else float("nan")
        frequency = (
            finite_frequency.mean().item()
            if finite_frequency.numel()
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

    return {
        "dataset_type": data.get("dataset_type", "test"),
        "states": estimates,
        "mse_per_trajectory": mse_per_trajectory,
        "position_mse_per_trajectory": pos_mse_per_trajectory,
        "velocity_mse_per_trajectory": vel_mse_per_trajectory,
        "class_id_per_trajectory": class_ids,
        "class_ids": unique_classes,
        "class_trace_ratios": torch.tensor(class_ratios),
        "class_frequencies": torch.tensor(class_frequencies),
        "class_mse": torch.tensor(class_mse),
        "class_position_mse": torch.tensor(class_pos_mse),
        "class_velocity_mse": torch.tensor(class_vel_mse),
        "class_metrics": class_metrics,
        "mean_mse": mse_per_trajectory.mean().item(),
        "mean_position_mse": pos_mse_per_trajectory.mean().item(),
        "mean_velocity_mse": vel_mse_per_trajectory.mean().item(),
    }


def print_metrics(result, prefix):
    dataset_name = result["dataset_type"]
    for class_id, metrics in result["class_metrics"].items():
        print(
            f"[{prefix} {dataset_name} class {class_id}] "
            f"ratio={metrics['trace_ratio']:.6g}, "
            f"frequency={metrics['frequency_cycles_per_trajectory']:.6g}, "
            f"MSE={metrics['mse']:.6f}, "
            f"position={metrics['position_mse']:.6f}, "
            f"velocity={metrics['velocity_mse']:.6f}"
        )
    print(
        f"[{prefix} {dataset_name} overall] MSE={result['mean_mse']:.6f}, "
        f"position={result['mean_position_mse']:.6f}, "
        f"velocity={result['mean_velocity_mse']:.6f}"
    )


def save_results(path, method, config, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"method": method, "config": config, "datasets": results}, path
    )
    print(f"saved {method} results to {path}")
