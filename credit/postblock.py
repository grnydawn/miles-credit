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

        # negative tracer fixer
        if post_conf['tracer_fixer']['activate']:
            opt = tracer_fixer(post_conf)
            self.operations.append(opt)

    def forward(self, x):
        for op in self.operations:
            x = op(x)
        return x

class tracer_fixer(nn.Module):
    '''
    This class non-negative tracers by replacing their negative values to zero.
    Modification is done witohut making copies.
    '''
    def __init__(self, post_conf):
        super().__init__()
        
        self.tracer_indices = conf['model']['post_conf']['tracer_fixer']['tracer_inds']
        
    def forward(self, x):
        # negtive tracer correction
        for i_var in self.tracer_indices
            # y_pred is channel first: (batch, var, time, lat, lon)
            tracer_vals = x["y_pred"][:, i_var, ...]
            
            # modify `x["y_pred"][:, i_var, ...]` in-place
            tracer_vals[tracer_vals<0] = 0
            
        return x

# class global_mass_fixer(nn.Module):
#     '''
#     '''
#     def __init__(self, post_conf):
#         super().__init__()

#     def forward(self, x):
#         x_input = x['x']
#         y_pred = x["y_pred"]
#         ##

# class global_energy_fixer(nn.Module):
#     '''
#     '''
#     def __init__(self, post_conf):
#         super().__init__()

#     def forward(self, x):
#         x_input = x['x']
#         y_pred = x["y_pred"]
#         ##

        
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
        x = x["y_pred"]
        return self.additional_layer(x)
    
if __name__ == "__main__":
    image_width = 100
    conf = {"post_conf": {"use_skebs": True, "image_width": image_width}}

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    y_pred = postblock(input_tensor)
    print("Predicted shape:", y_pred.shape)


