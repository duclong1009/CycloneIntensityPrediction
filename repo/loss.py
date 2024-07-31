import torch
import torch.nn as nn 

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # Calculate the mean squared error
        mse_loss = (y_true - y_pred) ** 2
        
        # Compute weights (higher for larger ground truth values)
        weights = torch.abs(y_true + 1e-3)
        
        # Apply weights
        weighted_loss = mse_loss * weights
        
        # Return the mean weighted loss
        return weighted_loss.mean()
    

