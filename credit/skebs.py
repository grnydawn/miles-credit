import torch
from torch import nn

# placeholder module for skebs, while software is fully engineered
# and tested

class SKEBS_module(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.image_width = conf['model']['image_width']
        final_layer_size = self.image_width
        self.additional_layer = nn.Linear(final_layer_size, final_layer_size)#.to(self.device) # Example: another layer
    
    def forward(self, x):
        return self.additional_layer(x)
    
if __name__ == "__main__":
    image_width = 100
    conf = {"model": {"image_width": image_width}}

    input_tensor = torch.randn(image_width)
    skebs = SKEBS_module(conf)

    y_pred = skebs(input_tensor)
    print("Predicted shape:", y_pred.shape)


