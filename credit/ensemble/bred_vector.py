import torch


def generate_bred_vectors(
        x_batch,
        model,
        x_forcing_batch=None,
        num_cycles=5,
        perturbation_size=0.01,
        epsilon=0.01,
        flag_clamp=False,
        clamp_min=None,
        clamp_max=None):
    """
    Generate bred vectors and initialize initial conditions for the given batch.

    Args:
        x_batch (torch.Tensor): The input batch.
        batch (dict): A dictionary containing additional batch data.
        model (nn.Module): The model used for predictions.
        num_cycles (int): Number of perturbation cycles.
        perturbation_size (float): Magnitude of initial perturbations.
        epsilon (float): Scaling factor for bred vectors.
        flag_clamp (bool, optional): Whether to clamp inputs. Defaults to False.
        clamp_min (float, optional): Minimum clamp value. Required if flag_clamp is True.
        clamp_max (float, optional): Maximum clamp value. Required if flag_clamp is True.

    Returns:
        list[torch.Tensor]: List of initial conditions generated using bred vectors.
    """
    bred_vectors = []
    for _ in range(num_cycles):
        # Create initial perturbation for entire batch
        delta_x0 = perturbation_size * torch.randn_like(x_batch)
        x_perturbed = x_batch.clone() + delta_x0

        # Run both unperturbed and perturbed forecasts
        x_unperturbed = x_batch.clone()

        if flag_clamp:
            x_unperturbed = torch.clamp(x_unperturbed, min=clamp_min, max=clamp_max)
            x_perturbed = torch.clamp(x_perturbed, min=clamp_min, max=clamp_max)

        # Batch predictions
        x_unperturbed_pred = model(x_unperturbed)
        x_perturbed_pred = model(x_perturbed)

        # Add forcing and static variables if present in batch
        if x_forcing_batch is not None:
            device = x_unperturbed_pred.device
            x_forcing_batch = x_forcing_batch.to(device)
            x_unperturbed_pred = torch.cat((x_unperturbed_pred, x_forcing_batch), dim=1)
            x_perturbed_pred = torch.cat((x_perturbed_pred, x_forcing_batch), dim=1)

        # Compute bred vectors
        delta_x = x_perturbed_pred - x_unperturbed_pred
        norm = torch.norm(delta_x, p=2, dim=2, keepdim=True)  # Calculate norm across channels
        delta_x_rescaled = epsilon * delta_x / (1e-8 + norm)
        bred_vectors.append(delta_x_rescaled)

    # Initialize ensemble members for the entire batch
    initial_conditions = [x_batch.clone() + bv for bv in bred_vectors]
    return initial_conditions
