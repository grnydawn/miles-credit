import torch
from credit.postBlock import PostBlock
from credit.postBlock import SKEBS
    
def test_SKEBS_rand():
    image_width = 100
    conf = {"post_conf": {"use_skebs": True, "image_width": image_width}}

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    y_pred = postblock(input_tensor)

    assert y_pred.shape == input_tensor.shape

def test_SKEBS_era5():
    """
    todo after implementation
    """
    pass

