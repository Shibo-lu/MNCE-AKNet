# -*- coding: utf-8 -*-
"""Differentiable KalmanNet recursion for the current 6D/4D tracking model."""

import torch
import torch.nn.functional as F


def acceleration_at(accelerations, time_index):
    return (
        accelerations[:, time_index]
        if accelerations.dim() == 3
        else accelerations
    )


def measurement_function(states):
    radius = torch.sqrt(states[:, :3].square().sum(dim=1) + 1e-8)
    return torch.stack(
        [radius, states[:, 3], states[:, 4], states[:, 5]], dim=1
    )


def measurement_jacobian(states):
    batch_size = states.size(0)
    radius = torch.sqrt(states[:, :3].square().sum(dim=1) + 1e-8)
    jacobian = states.new_zeros(batch_size, 4, 6)
    jacobian[:, 0, :3] = states[:, :3] / radius.unsqueeze(1)
    jacobian[:, 1, 3] = 1.0
    jacobian[:, 2, 4] = 1.0
    jacobian[:, 3, 5] = 1.0
    return jacobian


class KalmanNetFilter:
    def __init__(self, transition, control, device):
        self.transition = transition.to(device)
        self.control = control.to(device)
        self.device = torch.device(device)

    def rollout(
        self,
        model,
        measurements,
        accelerations,
        initial_states,
        query_hidden=False,
        return_measurement_predictions=False,
    ):
        """Run the paper recursion; inputs use [batch, time, feature]."""
        batch_size, sequence_length, _ = measurements.shape
        if sequence_length < 2:
            raise ValueError("A filtering sequence must contain at least two steps.")
        model.initialize_hidden(
            batch_size, query=query_hidden, device=measurements.device
        )

        state_post = initial_states
        estimates = [state_post]
        next_measurement_predictions = []
        first_step = True
        state_post_past = None
        state_predict_past = None
        observation_past = None
        transition_jacobian = self.transition.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Time zero is the known filter initialization, as in the source code.
        for time_index in range(1, sequence_length):
            observation = measurements[:, time_index]
            control_input = acceleration_at(accelerations, time_index - 1)
            state_predict = (
                state_post @ self.transition.T
                + control_input @ self.control.T
            )

            if first_step:
                state_post_past = state_post.detach().clone()
                state_predict_past = state_predict.detach().clone()
                observation_past = observation.detach().clone()

            predicted_measurement = measurement_function(state_predict)
            residual = observation - predicted_measurement
            state_innovation = state_post_past - state_predict_past
            state_difference = state_post - state_post_past
            measurement_difference = observation - observation_past
            observation_jacobian = measurement_jacobian(state_predict)

            # Preserve the positional feature order used by filter.py in the
            # authors' repository.
            kalman_gain = model(
                measurement_difference,
                residual,
                state_difference,
                state_innovation,
                transition_jacobian,
                observation_jacobian,
            )
            state_post_new = state_predict + (
                kalman_gain @ residual.unsqueeze(-1)
            ).squeeze(-1)
            estimates.append(state_post_new)

            # The source semi-supervised loss compares the one-step-ahead
            # predicted measurement with the next observed measurement.
            if return_measurement_predictions and time_index < sequence_length - 1:
                next_control = acceleration_at(accelerations, time_index)
                next_state_predict = (
                    state_post_new @ self.transition.T
                    + next_control @ self.control.T
                )
                next_measurement_predictions.append(
                    measurement_function(next_state_predict)
                )

            first_step = False
            state_predict_past = state_predict.detach().clone()
            state_post_past = state_post.detach().clone()
            observation_past = observation.detach().clone()
            # The original implementation truncates state recursion gradients.
            state_post = state_post_new.detach().clone()

        output = {"states": torch.stack(estimates, dim=1)}
        if return_measurement_predictions:
            output["next_measurement_predictions"] = torch.stack(
                next_measurement_predictions, dim=1
            )
            output["next_measurement_targets"] = measurements[:, 2:]
        return output

    def supervised_loss(
        self, model, states, measurements, accelerations, initial_states
    ):
        estimates = self.rollout(
            model, measurements, accelerations, initial_states
        )["states"]
        return F.mse_loss(estimates[:, 1:], states[:, 1:])

    def measurement_prediction_loss(
        self, model, measurements, accelerations, initial_states
    ):
        if measurements.size(1) < 3:
            raise ValueError(
                "Semi-supervised adaptation needs at least three time steps."
            )
        output = self.rollout(
            model,
            measurements,
            accelerations,
            initial_states,
            return_measurement_predictions=True,
        )
        return F.mse_loss(
            output["next_measurement_predictions"],
            output["next_measurement_targets"],
        )
