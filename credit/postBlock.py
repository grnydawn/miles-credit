import torch
from torch import nn

import logging

class PostBlock(nn.Module):
    def __init__(self, 
                 post_conf):
        """
            post_conf: dictionary with config options for PostBlock.
                       if post_conf is not specified in config, 
                       defaults are set in the parser

            This class is a wrapper for all post-model operations such as
            SKEBS, lapplacian pole filtering, mass correction, core diffusion
        """
        super().__init__()

        self.operations = nn.ModuleList()
        if post_conf["use_skebs"]:
            logging.info("using SKEBS")
            self.operations.append(SKEBS(post_conf))

    def forward(self, x):
        for op in self.operations:
            x = op(x)
        return x
    
class SKEBS(nn.Module):
    """
        post_conf: dictionary with config options for PostBlock.
                    if post_conf is not specified in config, 
                    defaults are set in the parser

        This class is currently a placeholder for SKEBS
    """
    def __init__(self, post_conf):
        super().__init__()
        self.image_width = post_conf['image_width']
        final_layer_size = self.image_width
        self.additional_layer = nn.Linear(final_layer_size, final_layer_size)#.to(self.device) # Example: another layer
    
    def forward(self, x):
        return self.additional_layer(x)
    
if __name__ == "__main__":
    image_width = 100
    conf = {"post_conf": {"use_skebs": True, "image_width": image_width}}

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    y_pred = postblock(input_tensor)
    print("Predicted shape:", y_pred.shape)


