import torch
from credit.postblock import PostBlock
from credit.postblock import SKEBS
    
def test_SKEBS_rand():
    image_width = 100
    conf = {"post_conf": {"use_skebs": True, "image_width": image_width,}}
    conf['post_conf'].setdefault('tracer_fixer', {'activate': False});

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    input_dict = {"y_pred": input_tensor}

    y_pred = postblock(input_dict)

    assert y_pred.shape == input_tensor.shape

def test_SKEBS_era5():
    """
    todo after implementation
    """
    pass

