"""Generate task-structured trajectories for training and testing.

Each task contains M trajectories drawn from one fixed Gaussian Q/R pair.
Different tasks use different covariance shapes/scales while preserving the
requested trace(Q) / trace(R) class exactly.

The test generator additionally creates three separately saved datasets:
S0 uses fixed unseen trace ratios, S1 inserts a long abrupt-noise interval into
each S0 class, and the five S3 classes use different log-scale covariance
variation frequencies.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from Dynamics import AcceleratedMovementModel


TRACE_RATIOS = (0.01, 0.1, 1.0, 10.0, 100.0)
TEST_RATIO_MIN = 0.02
TEST_RATIO_MAX = 99.0
TEST_CLASSES = 5
S3_FREQUENCIES = (0.25, 0.5, 1.0, 2.0, 4.0)
S3_LOG10_AMPLITUDE = 0.5


def _sample_diagonal_covariance(trace_value, dim, rng, min_weight=0.25):
    """Sample a positive diagonal covariance with an exact prescribed trace."""
    if trace_value == 0.0:
        return np.zeros((dim, dim), dtype=np.float32)
    weights = rng.dirichlet(np.full(dim, 2.0)) + min_weight / dim
    weights = weights / weights.sum()
    return np.diag(trace_value * weights).astype(np.float32)


def _observe(states):
    ranges = np.linalg.norm(states[:, :3], axis=1, keepdims=True)
    return np.concatenate((ranges, states[:, 3:]), axis=1)


def _sample_task_covariances(
    ratio,
    rng,
    base_process_trace,
    scale_log_range,
    fixed_scale=None,
):
    # Training uses a random total process scale. Test generation can pass
    # fixed_scale=1 so every S0/S1 covariance has the same process-noise trace.
    scale = (
        10.0 ** rng.uniform(-scale_log_range, scale_log_range)
        if fixed_scale is None
        else float(fixed_scale)
    )
    if scale <= 0.0:
        raise ValueError("Covariance scale must be positive.")
    q_trace = base_process_trace * scale
    r_trace = q_trace / ratio
    q_cov = _sample_diagonal_covariance(q_trace, 6, rng)
    r_cov = _sample_diagonal_covariance(r_trace, 4, rng)
    return q_cov, r_cov


def _sample_unseen_sorted_ratios(
    count,
    rng,
    excluded,
    low=TEST_RATIO_MIN,
    high=TEST_RATIO_MAX,
    relative_gap=0.02,
):
    """Draw sorted log-stratified ratios that are separated from exclusions."""
    if not 0.0 < low < high <= 100.0:
        raise ValueError("Ratio bounds must satisfy 0 < low < high <= 100.")
    excluded = [float(value) for value in excluded]
    edges = np.geomspace(low, high, count + 1)
    ratios = []
    for index in range(count):
        for _ in range(10000):
            candidate = 10.0 ** rng.uniform(
                np.log10(edges[index]), np.log10(edges[index + 1])
            )
            references = excluded + ratios
            if all(
                abs(candidate - reference)
                > relative_gap * max(abs(reference), 1e-2)
                for reference in references
            ):
                ratios.append(float(candidate))
                break
        else:
            raise RuntimeError("Could not sample a trace ratio distinct from exclusions.")
    return tuple(sorted(ratios))


def _diagonal_sequence(diagonal_values):
    diagonal_values = np.asarray(diagonal_values, dtype=np.float32)
    sequence = np.zeros(
        diagonal_values.shape + (diagonal_values.shape[-1],), dtype=np.float32
    )
    indices = np.arange(diagonal_values.shape[-1])
    sequence[..., indices, indices] = diagonal_values
    return sequence


def _simulate_trajectory(f_mat, b_mat, q_sequence, r_sequence, rng):
    """Simulate one trajectory using time-indexed diagonal Q and R matrices."""
    num_steps, state_dim, _ = q_sequence.shape
    obs_dim = r_sequence.shape[-1]
    acceleration = rng.uniform(-0.05, 0.2, b_mat.shape[1]).astype(np.float32)
    initial_state = np.array(
        [0.0, 2.0, 3.0, 0.5, 0.5, 0.5], dtype=np.float32
    )
    states = np.zeros((num_steps, state_dim), dtype=np.float32)
    states[0] = initial_state
    q_std = np.sqrt(np.maximum(np.diagonal(q_sequence, axis1=-2, axis2=-1), 0.0))
    for t in range(num_steps - 1):
        process_noise = rng.normal(0.0, q_std[t], state_dim).astype(np.float32)
        states[t + 1] = (
            f_mat @ states[t] + b_mat @ acceleration + process_noise
        )

    clean_measurements = _observe(states).astype(np.float32)
    r_std = np.sqrt(np.maximum(np.diagonal(r_sequence, axis1=-2, axis2=-1), 0.0))
    measurement_noise = rng.normal(
        0.0, 1.0, size=(num_steps, obs_dim)
    ).astype(np.float32) * r_std
    return states, clean_measurements + measurement_noise, acceleration


def _assemble_test_dataset(
    dataset_type,
    q_sequences,
    r_sequences,
    class_ratios,
    f_mat,
    b_mat,
    rng,
    config,
    extra_fields=None,
):
    """Simulate and package schedules shaped [class, trajectory, time, d, d]."""
    num_classes, trajectories_per_class, num_steps = q_sequences.shape[:3]
    states = []
    measurements = []
    accelerations = []
    for class_index in range(num_classes):
        class_states = []
        class_measurements = []
        class_accelerations = []
        for trajectory_index in range(trajectories_per_class):
            state, measurement, acceleration = _simulate_trajectory(
                f_mat,
                b_mat,
                q_sequences[class_index, trajectory_index],
                r_sequences[class_index, trajectory_index],
                rng,
            )
            class_states.append(state)
            class_measurements.append(measurement)
            class_accelerations.append(acceleration)
        states.append(np.stack(class_states))
        measurements.append(np.stack(class_measurements))
        accelerations.append(np.stack(class_accelerations))
        print(
            f"generated {dataset_type} class {class_index + 1}/{num_classes} "
            f"(class ratio={class_ratios[class_index]:.6g})"
        )

    trace_q = np.trace(q_sequences, axis1=-2, axis2=-1)
    trace_r = np.trace(r_sequences, axis1=-2, axis2=-1)
    data = {
        "dataset_type": dataset_type,
        "true_states": torch.from_numpy(np.stack(states)),
        "measurements": torch.from_numpy(np.stack(measurements)),
        "acc_true": torch.from_numpy(np.stack(accelerations)),
        "Q": torch.from_numpy(q_sequences.astype(np.float32)),
        "R": torch.from_numpy(r_sequences.astype(np.float32)),
        "trace_ratio_time": torch.from_numpy(
            (trace_q / trace_r).astype(np.float32)
        ),
        "class_trace_ratio": torch.tensor(class_ratios, dtype=torch.float32),
        "class_id": torch.arange(num_classes, dtype=torch.long),
        "trajectory_class_id": torch.arange(num_classes, dtype=torch.long)
        .unsqueeze(1)
        .expand(num_classes, trajectories_per_class)
        .clone(),
        "config": config,
    }
    if extra_fields:
        data.update(extra_fields)
    return data


def generate_time_varying_test_datasets(
    output_dir,
    trajectories_per_class=64,
    num_steps=256,
    dt=0.1,
    seed=2026,
    base_process_trace=0.06,
    scale_log_range=0.5,
    smooth_log10_amplitude=S3_LOG10_AMPLITUDE,
):
    """Generate and separately save S0 fixed, S1 abrupt, and S3 smooth tests."""
    if trajectories_per_class < 1:
        raise ValueError("trajectories_per_class must be positive.")
    if num_steps < 8:
        raise ValueError("num_steps must be at least 8 for abrupt intervals.")
    if smooth_log10_amplitude <= 0.0:
        raise ValueError("smooth_log10_amplitude must be positive.")
    if len(S3_FREQUENCIES) != TEST_CLASSES or any(
        frequency <= 0.0 for frequency in S3_FREQUENCIES
    ):
        raise ValueError("S3_FREQUENCIES must contain five positive values.")

    rng = np.random.default_rng(seed)
    dynamics = AcceleratedMovementModel(
        torch.tensor(dt, dtype=torch.float32), device="cpu"
    )
    _, _, _, f_mat, _, b_mat = dynamics.get_dynamics()
    f_mat = f_mat.cpu().numpy()
    b_mat = b_mat.cpu().numpy()

    s0_ratios = _sample_unseen_sorted_ratios(
        TEST_CLASSES, rng, excluded=TRACE_RATIOS
    )
    s1_ratios = _sample_unseen_sorted_ratios(
        TEST_CLASSES, rng, excluded=TRACE_RATIOS + s0_ratios
    )

    s0_covariances = [
        _sample_task_covariances(
            ratio,
            rng,
            base_process_trace,
            scale_log_range,
            fixed_scale=1.0,
        )
        for ratio in s0_ratios
    ]
    s1_covariances = [
        _sample_task_covariances(
            ratio,
            rng,
            base_process_trace,
            scale_log_range,
            fixed_scale=1.0,
        )
        for ratio in s1_ratios
    ]
    q_s0 = np.empty(
        (TEST_CLASSES, trajectories_per_class, num_steps, 6, 6),
        dtype=np.float32,
    )
    r_s0 = np.empty(
        (TEST_CLASSES, trajectories_per_class, num_steps, 4, 4),
        dtype=np.float32,
    )
    for class_index, (q_cov, r_cov) in enumerate(s0_covariances):
        q_s0[class_index] = q_cov
        r_s0[class_index] = r_cov

    common_config = {
        "trajectories_per_class": trajectories_per_class,
        "num_classes": TEST_CLASSES,
        "num_steps": num_steps,
        "dt": dt,
        "seed": seed,
        "base_process_trace": base_process_trace,
        "scale_log_range": scale_log_range,
        "fixed_process_trace_s0_s1": base_process_trace,
        "training_trace_ratios": TRACE_RATIOS,
        "ratio_sampling_range": (TEST_RATIO_MIN, TEST_RATIO_MAX),
    }
    s0_data = _assemble_test_dataset(
        "S0_fixed",
        q_s0,
        r_s0,
        s0_ratios,
        f_mat,
        b_mat,
        rng,
        {**common_config, "class_trace_ratios": s0_ratios},
        extra_fields={
            "class_process_trace": torch.full(
                (TEST_CLASSES,), base_process_trace, dtype=torch.float32
            ),
            "class_measurement_trace": torch.tensor(
                [base_process_trace / ratio for ratio in s0_ratios],
                dtype=torch.float32,
            ),
        },
    )

    q_s1 = q_s0.copy()
    r_s1 = r_s0.copy()
    change_start = np.empty(
        (TEST_CLASSES, trajectories_per_class), dtype=np.int64
    )
    change_end = np.empty_like(change_start)
    minimum_duration = num_steps // 2 + 1
    edge_margin = max(1, min(8, (num_steps - minimum_duration) // 2))
    maximum_duration = num_steps - 2 * edge_margin
    for class_index, (jump_q, jump_r) in enumerate(s1_covariances):
        for trajectory_index in range(trajectories_per_class):
            duration = int(
                rng.integers(minimum_duration, maximum_duration + 1)
            )
            latest_start = num_steps - edge_margin - duration
            start = int(rng.integers(edge_margin, latest_start + 1))
            end = start + duration
            q_s1[class_index, trajectory_index, start:end] = jump_q
            r_s1[class_index, trajectory_index, start:end] = jump_r
            change_start[class_index, trajectory_index] = start
            change_end[class_index, trajectory_index] = end
    s1_data = _assemble_test_dataset(
        "S1_abrupt",
        q_s1,
        r_s1,
        s1_ratios,
        f_mat,
        b_mat,
        rng,
        {
            **common_config,
            "base_class_trace_ratios": s0_ratios,
            "abrupt_class_trace_ratios": s1_ratios,
            "minimum_abrupt_duration": minimum_duration,
        },
        extra_fields={
            "base_class_trace_ratio": torch.tensor(
                s0_ratios, dtype=torch.float32
            ),
            "base_class_process_trace": torch.full(
                (TEST_CLASSES,), base_process_trace, dtype=torch.float32
            ),
            "abrupt_class_process_trace": torch.full(
                (TEST_CLASSES,), base_process_trace, dtype=torch.float32
            ),
            "base_class_measurement_trace": torch.tensor(
                [base_process_trace / ratio for ratio in s0_ratios],
                dtype=torch.float32,
            ),
            "abrupt_class_measurement_trace": torch.tensor(
                [base_process_trace / ratio for ratio in s1_ratios],
                dtype=torch.float32,
            ),
            "change_start": torch.from_numpy(change_start),
            "change_end": torch.from_numpy(change_end),
        },
    )

    q_s3 = np.empty_like(q_s0)
    r_s3 = np.empty_like(r_s0)
    smooth_phase = np.zeros(
        (TEST_CLASSES, trajectories_per_class), dtype=np.float32
    )
    time = np.arange(num_steps, dtype=np.float64)
    # All S3 classes share the same reference covariance and amplitude. Only
    # the number of covariance cycles per trajectory changes between classes.
    reference_class_index = TEST_CLASSES // 2
    reference_q, reference_r = s0_covariances[reference_class_index]
    reference_ratio = s0_ratios[reference_class_index]
    for class_index, frequency in enumerate(S3_FREQUENCIES):
        angle = 2.0 * np.pi * frequency * time / max(1, num_steps - 1)
        q_scale = 10.0 ** (smooth_log10_amplitude * np.sin(angle))
        # A quarter-cycle offset prevents Q and R from changing identically.
        r_scale = 10.0 ** (
            smooth_log10_amplitude * np.sin(angle + 0.5 * np.pi)
        )
        q_sequence = (
            q_scale[:, None, None] * reference_q[None, :, :]
        ).astype(np.float32)
        r_sequence = (
            r_scale[:, None, None] * reference_r[None, :, :]
        ).astype(np.float32)
        for trajectory_index in range(trajectories_per_class):
            q_s3[class_index, trajectory_index] = q_sequence
            r_s3[class_index, trajectory_index] = r_sequence
    s3_class_ratios = (reference_ratio,) * TEST_CLASSES
    class_period_steps = tuple(
        float(num_steps - 1) / frequency for frequency in S3_FREQUENCIES
    )
    s3_data = _assemble_test_dataset(
        "S3_smooth",
        q_s3,
        r_s3,
        s3_class_ratios,
        f_mat,
        b_mat,
        rng,
        {
            **common_config,
            "reference_trace_ratio": reference_ratio,
            "reference_s0_class": reference_class_index,
            "smooth_frequencies_cycles_per_trajectory": S3_FREQUENCIES,
            "smooth_period_steps": class_period_steps,
            "smooth_log10_amplitude": smooth_log10_amplitude,
            "smooth_schedule": "base-10 log-scale sinusoidal modulation",
        },
        extra_fields={
            "reference_trace_ratio": torch.tensor(
                reference_ratio, dtype=torch.float32
            ),
            "class_frequency": torch.tensor(
                S3_FREQUENCIES, dtype=torch.float32
            ),
            "class_period_steps": torch.tensor(
                class_period_steps, dtype=torch.float32
            ),
            "smooth_phase": torch.from_numpy(smooth_phase),
        },
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "S0": (output_dir / "maml_test_S0_fixed.pt", s0_data),
        "S1": (output_dir / "maml_test_S1_abrupt.pt", s1_data),
        "S3": (output_dir / "maml_test_S3_smooth.pt", s3_data),
    }
    for name, (path, data) in outputs.items():
        torch.save(data, path)
        print(f"saved {name} test dataset to {path}")
    return {name: path for name, (path, _) in outputs.items()}


def generate_meta_dataset(
    output_path,
    tasks_per_class,
    trajectories_per_task,
    num_steps=256,
    dt=0.1,
    seed=42,
    base_process_trace=0.06,
    scale_log_range=0.5,
):
    if tasks_per_class < 1:
        raise ValueError("tasks_per_class (N) must be at least 1.")
    if trajectories_per_task < 2:
        raise ValueError("trajectories_per_task (M) must be at least 2.")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")
    if base_process_trace <= 0:
        raise ValueError("base_process_trace must be positive.")

    rng = np.random.default_rng(seed)
    dynamics = AcceleratedMovementModel(
        torch.tensor(dt, dtype=torch.float32), device="cpu"
    )
    state_dim, obs_dim, acc_dim, f_mat, _, b_mat = dynamics.get_dynamics()
    f_mat = f_mat.cpu().numpy()
    b_mat = b_mat.cpu().numpy()
    initial_state = np.array(
        [0.0, 2.0, 3.0, 0.5, 0.5, 0.5], dtype=np.float32
    )

    task_states = []
    task_measurements = []
    task_accelerations = []
    task_q = []
    task_r = []
    task_ratios = []
    task_classes = []

    for class_index, ratio in enumerate(TRACE_RATIOS):
        for class_task_index in range(tasks_per_class):
            q_cov, r_cov = _sample_task_covariances(
                ratio, rng, base_process_trace, scale_log_range
            )
            actual_ratio = float(np.trace(q_cov) / np.trace(r_cov))
            if not np.isclose(actual_ratio, ratio, rtol=1e-5, atol=1e-7):
                raise RuntimeError(
                    f"Generated trace ratio {actual_ratio} does not match {ratio}."
                )
            states_for_task = []
            measurements_for_task = []
            accelerations_for_task = []

            for _ in range(trajectories_per_task):
                acceleration = rng.uniform(-0.05, 0.2, acc_dim).astype(np.float32)
                states = np.zeros((num_steps, state_dim), dtype=np.float32)
                states[0] = initial_state
                for t in range(num_steps - 1):
                    process_noise = rng.multivariate_normal(
                        np.zeros(state_dim), q_cov, check_valid="ignore"
                    ).astype(np.float32)
                    states[t + 1] = (
                        f_mat @ states[t] + b_mat @ acceleration + process_noise
                    )

                measurement_noise = rng.multivariate_normal(
                    np.zeros(obs_dim),
                    r_cov,
                    size=num_steps,
                    check_valid="ignore",
                ).astype(np.float32)
                measurements = _observe(states).astype(np.float32) + measurement_noise
                states_for_task.append(states)
                measurements_for_task.append(measurements)
                accelerations_for_task.append(acceleration)

            task_states.append(np.stack(states_for_task))
            task_measurements.append(np.stack(measurements_for_task))
            task_accelerations.append(np.stack(accelerations_for_task))
            task_q.append(np.repeat(q_cov[None], num_steps, axis=0))
            task_r.append(np.repeat(r_cov[None], num_steps, axis=0))
            task_ratios.append(ratio)
            task_classes.append(class_index)
            print(
                f"generated task {len(task_states)}/{len(TRACE_RATIOS) * tasks_per_class} "
                f"(ratio={ratio:g}, class task={class_task_index + 1}/{tasks_per_class})"
            )

    data = {
        # Leading dimensions are [task, trajectory, time, ...].
        "true_states": torch.from_numpy(np.stack(task_states)),
        "measurements": torch.from_numpy(np.stack(task_measurements)),
        "acc_true": torch.from_numpy(np.stack(task_accelerations)),
        # Q/R are task-level distributions, shape [task, time, dim, dim].
        "Q": torch.from_numpy(np.stack(task_q)),
        "R": torch.from_numpy(np.stack(task_r)),
        "trace_ratio": torch.tensor(task_ratios, dtype=torch.float32),
        "class_id": torch.tensor(task_classes, dtype=torch.long),
        "task_id": torch.arange(len(task_states), dtype=torch.long),
        "config": {
            "tasks_per_class": tasks_per_class,
            "trajectories_per_task": trajectories_per_task,
            "num_steps": num_steps,
            "dt": dt,
            "seed": seed,
            "base_process_trace": base_process_trace,
            "scale_log_range": scale_log_range,
            "trace_ratios": TRACE_RATIOS,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(
        f"saved {len(task_states)} tasks x {trajectories_per_task} trajectories "
        f"to {output_path}"
    )
    return data


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("train", "test", "all"),
        default="test",
        help="Generate the MAML train set, the three test sets, or both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "traj_dataset1" / "maml_meta_train.pt",
    )
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        default=root / "traj_dataset1",
    )
    parser.add_argument("--tasks-per-class", "-N", type=int, default=20)
    parser.add_argument("--trajectories-per-task", "-M", type=int, default=144)
    parser.add_argument("--test-trajectories-per-class", type=int, default=64)
    parser.add_argument(
        "--smooth-log10-amplitude",
        type=float,
        default=S3_LOG10_AMPLITUDE,
        help=(
            "S3 sinusoidal amplitude in log10 covariance scale. The default "
            "0.5 gives covariance multipliers from about 0.316 to 3.162."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-process-trace", type=float, default=0.2)
    parser.add_argument(
        "--scale-log-range",
        type=float,
        default=0.5,
        help="Task trace multiplier is sampled log-uniformly from 10^-x to 10^x.",
    )
    args = parser.parse_args()
    if args.mode in ("train", "all"):
        generate_meta_dataset(
            output_path=args.output,
            tasks_per_class=args.tasks_per_class,
            trajectories_per_task=args.trajectories_per_task,
            num_steps=args.num_steps,
            dt=args.dt,
            seed=args.seed,
            base_process_trace=args.base_process_trace,
            scale_log_range=args.scale_log_range,
        )
    if args.mode in ("test", "all"):
        generate_time_varying_test_datasets(
            output_dir=args.test_output_dir,
            trajectories_per_class=args.test_trajectories_per_class,
            num_steps=args.num_steps,
            dt=args.dt,
            seed=args.seed + 10000,
            base_process_trace=args.base_process_trace,
            scale_log_range=args.scale_log_range,
            smooth_log10_amplitude=args.smooth_log10_amplitude,
        )


if __name__ == "__main__":
    main()
