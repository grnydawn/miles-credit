import torch

def anomaly_correlation_coefficient(pred, true):
    
    pred = pred.float()
    true = true.float()
    
    B, C, H, W = pred.size()

    # Flatten the spatial dimensions
    pred_flat = pred.view(B, C, -1)
    true_flat = true.view(B, C, -1)

    # Mean over spatial dimensions
    pred_mean = torch.mean(pred_flat, dim=-1, keepdim=True)
    true_mean = torch.mean(true_flat, dim=-1, keepdim=True)

    # Anomaly calculation
    pred_anomaly = pred_flat - pred_mean
    true_anomaly = true_flat - true_mean

    # Covariance matrix
    covariance_matrix = torch.bmm(pred_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Variance terms
    pred_var = torch.bmm(pred_anomaly, pred_anomaly.transpose(1, 2)) / (H * W - 1)
    true_var = torch.bmm(true_anomaly, true_anomaly.transpose(1, 2)) / (H * W - 1)

    # Anomaly Correlation Coefficient
    acc_numerator = torch.einsum('bii->b', covariance_matrix).sum()
    acc_denominator = torch.sqrt(torch.einsum('bii->b', pred_var).sum() * torch.einsum('bii->b', true_var).sum())

    # Avoid division by zero
    epsilon = 1e-8
    acc = acc_numerator / (acc_denominator + epsilon)

    return acc.item()