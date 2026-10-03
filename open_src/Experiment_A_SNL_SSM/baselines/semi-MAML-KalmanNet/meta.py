# -*- coding: utf-8 -*-
"""Paper-aligned MSG/FOMAML training with current-dataset tensor adapters."""

import copy
import math

import torch


FIXED_INITIAL_STATE = (0.0, 2.0, 3.0, 0.5, 0.5, 0.5)


def initial_states(states, use_true_x0):
    if use_true_x0:
        return states[:, 0]
    return states.new_tensor(FIXED_INITIAL_STATE).unsqueeze(0).expand(
        states.size(0), -1
    )


def task_weight(trace_ratio):
    """Paper/code weight derived from the process/measurement noise ratio."""
    ratio = max(float(trace_ratio), 1e-12)
    return 1.0 / (1.0 + abs(math.log10(ratio)))


def _select_trajectories(indices, batch_size):
    if indices.numel() == 0:
        raise ValueError("Cannot sample from an empty trajectory set.")
    if batch_size >= indices.numel():
        return indices
    order = torch.randperm(indices.numel())[:batch_size]
    return indices[order]


def task_batch(task, indices, batch_size, segment_len, device):
    chosen = _select_trajectories(indices, batch_size)
    states = task["true_states"][chosen, :segment_len].to(device)
    measurements = task["measurements"][chosen, :segment_len].to(device)
    accelerations = task["acc_true"][chosen]
    if accelerations.dim() == 3:
        accelerations = accelerations[:, :segment_len]
    return states, measurements, accelerations.to(device)


def split_task(num_trajectories, support_size, query_size):
    if support_size < 1 or query_size < 1:
        raise ValueError("support_size and query_size must be positive.")
    if support_size + query_size > num_trajectories:
        raise ValueError(
            f"Task has {num_trajectories} trajectories, but "
            f"support_size + query_size = {support_size + query_size}."
        )
    order = torch.randperm(num_trajectories)
    return order[:support_size], order[support_size : support_size + query_size]


def first_order_task_gradients(
    base_model,
    filter_runner,
    task,
    support_indices,
    query_indices,
    config,
    device,
):
    """Compute the paper's K weighted post-update query gradients for one task."""
    task_model = copy.deepcopy(base_model).to(device)
    task_model.train()
    inner_optimizer = torch.optim.SGD(
        task_model.parameters(), lr=config["inner_lr"]
    )
    gradient_sums = {
        name: torch.zeros_like(parameter)
        for name, parameter in task_model.named_parameters()
        if parameter.requires_grad
    }
    support_loss_value = 0.0
    query_loss_value = 0.0

    for update_index in range(config["inner_steps"]):
        states, measurements, accelerations = task_batch(
            task,
            support_indices,
            config["inner_batch_size"],
            config["segment_len"],
            device,
        )
        support_loss = filter_runner.supervised_loss(
            task_model,
            states,
            measurements,
            accelerations,
            initial_states(states, config["true_x0"]),
        )
        inner_optimizer.zero_grad(set_to_none=True)
        support_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            task_model.parameters(), config["grad_clip"]
        )
        inner_optimizer.step()
        support_loss_value += support_loss.detach().item()

        step_weights = config["msg_step_weights"]
        step_factor = step_weights[
            min(update_index, len(step_weights) - 1)
        ]
        gradient_factor = step_factor * task_weight(task["trace_ratio"])
        query_states, query_measurements, query_accelerations = task_batch(
            task,
            query_indices,
            config["query_batch_size"],
            config["segment_len"],
            device,
        )
        query_loss = filter_runner.supervised_loss(
            task_model,
            query_states,
            query_measurements,
            query_accelerations,
            initial_states(query_states, config["true_x0"]),
        )
        # Equation (21): use the query gradient at theta^k only. Clearing the
        # support gradient prevents it from contaminating G_{tau_b}^k.
        inner_optimizer.zero_grad(set_to_none=True)
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            task_model.parameters(), config["grad_clip"]
        )
        for name, parameter in task_model.named_parameters():
            if parameter.grad is not None:
                gradient_sums[name].add_(
                    parameter.grad.detach(), alpha=gradient_factor
                )
        query_loss_value += query_loss.detach().item()

    return {
        "gradients": gradient_sums,
        "support_loss": support_loss_value / config["inner_steps"],
        "query_loss": query_loss_value / config["inner_steps"],
    }


def meta_update(
    base_model,
    filter_runner,
    meta_optimizer,
    tasks,
    config,
    device,
    use_msg,
):
    """Perform one meta-batch update using the paper's two-phase schedule."""
    meta_optimizer.zero_grad(set_to_none=True)
    if not use_msg:
        return paper_second_phase_update(
            base_model,
            filter_runner,
            meta_optimizer,
            tasks,
            config,
            device,
        )

    accumulated = {
        name: torch.zeros_like(parameter)
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }
    support_total = 0.0
    query_total = 0.0
    for task in tasks:
        support_indices, query_indices = split_task(
            task["true_states"].size(0),
            config["support_size"],
            config["query_size"],
        )
        output = first_order_task_gradients(
            base_model,
            filter_runner,
            task,
            support_indices,
            query_indices,
            config,
            device,
        )
        for name, gradient in output["gradients"].items():
            accumulated[name].add_(gradient)
        support_total += output["support_loss"]
        query_total += output["query_loss"]

    scale = 1.0 / max(len(tasks), 1)
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = accumulated[name].mul(scale)
    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config["grad_clip"])
    meta_optimizer.step()
    return {
        "support_mse": support_total * scale,
        "query_mse": query_total * scale,
    }


def paper_second_phase_update(
    base_model,
    filter_runner,
    meta_optimizer,
    tasks,
    config,
    device,
):
    """Apply first-order MAML using the mean query gradient of every task."""
    support_total = 0.0
    task_models = []
    query_losses = []

    for task in tasks:
        support_indices, query_indices = split_task(
            task["true_states"].size(0),
            config["support_size"],
            config["query_size"],
        )
        task_model = copy.deepcopy(base_model).to(device)
        task_model.train()
        inner_optimizer = torch.optim.Adam(
            task_model.parameters(), lr=config["standard_inner_lr"]
        )
        for _ in range(config["inner_steps"]):
            states, measurements, accelerations = task_batch(
                task,
                support_indices,
                config["inner_batch_size"],
                config["segment_len"],
                device,
            )
            support_loss = filter_runner.supervised_loss(
                task_model,
                states,
                measurements,
                accelerations,
                initial_states(states, config["true_x0"]),
            )
            inner_optimizer.zero_grad(set_to_none=True)
            support_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                task_model.parameters(), config["grad_clip"]
            )
            inner_optimizer.step()
            support_total += support_loss.detach().item()

        # The first-order outer gradient must contain query information only.
        task_model.zero_grad(set_to_none=True)
        query_states, query_measurements, query_accelerations = task_batch(
            task,
            query_indices,
            config["query_batch_size"],
            config["segment_len"],
            device,
        )
        query_loss = filter_runner.supervised_loss(
            task_model,
            query_states,
            query_measurements,
            query_accelerations,
            initial_states(query_states, config["true_x0"]),
        )
        query_losses.append(query_loss)
        task_models.append(task_model)

    meta_loss = torch.stack(query_losses).mean()
    meta_loss.backward()

    # Equation (16) and Algorithm 1: aggregate the query gradients from all
    # tasks in the meta-batch. Because meta_loss is a mean, each task gradient
    # already carries the factor 1 / B.
    accumulated = {
        name: torch.zeros_like(parameter)
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }
    for task_model in task_models:
        for name, parameter in task_model.named_parameters():
            if parameter.grad is not None:
                accumulated[name].add_(parameter.grad.detach())

    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = accumulated[name]
    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config["grad_clip"])
    meta_optimizer.step()
    return {
        "support_mse": support_total
        / max(len(tasks) * config["inner_steps"], 1),
        "query_mse": meta_loss.detach().item(),
    }


def adapted_query_loss(
    base_model,
    filter_runner,
    task,
    config,
    device,
):
    """Validation loss after supervised few-shot adaptation."""
    support_indices, query_indices = split_task(
        task["true_states"].size(0),
        config["support_size"],
        config["query_size"],
    )
    task_model = copy.deepcopy(base_model).to(device)
    task_model.train()
    optimizer = torch.optim.SGD(task_model.parameters(), lr=config["inner_lr"])
    for _ in range(config["inner_steps"]):
        states, measurements, accelerations = task_batch(
            task,
            support_indices,
            config["inner_batch_size"],
            config["segment_len"],
            device,
        )
        loss = filter_runner.supervised_loss(
            task_model,
            states,
            measurements,
            accelerations,
            initial_states(states, config["true_x0"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model.parameters(), config["grad_clip"])
        optimizer.step()

    task_model.eval()
    states, measurements, accelerations = task_batch(
        task,
        query_indices,
        query_indices.numel(),
        config["segment_len"],
        device,
    )
    with torch.no_grad():
        query_loss = filter_runner.supervised_loss(
            task_model,
            states,
            measurements,
            accelerations,
            initial_states(states, config["true_x0"]),
        )
    return query_loss.item()
