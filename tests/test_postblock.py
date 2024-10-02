import torch
from credit.postblock import PostBlock
from credit.postblock import SKEBS
from credit.postblock import tracer_fixer
    
def test_SKEBS_rand():
    image_width = 100
    conf = {"post_conf": {"skebs": {'activate': True}, 
                          "model": {"image_width": image_width,}}}
    conf['post_conf'].setdefault('tracer_fixer', {'activate': False})

    input_tensor = torch.randn(image_width)
    postblock = PostBlock(**conf)
    assert any([isinstance(module, SKEBS) for module in postblock.modules()])

    input_dict = {"y_pred": input_tensor}

    y_pred = postblock(input_dict)

    assert y_pred.shape == input_tensor.shape

def test_tracer_fixer_rand():
    # conf keywords
    conf = {"post_conf": {"skebs": {'activate': False}}}
    conf['post_conf']['tracer_fixer'] = {'activate': True, 'denorm': False}
    conf['post_conf']['tracer_fixer']['tracer_inds'] = [0,]
    conf['post_conf']['tracer_fixer']['tracer_thres'] = [0,]

    # a random tensor with neg values
    input_tensor = -999*torch.randn((1, 1, 10, 10))

    # initialize postblock for 'tracer_fixer' only
    postblock = PostBlock(**conf)

    # verify that tracer_fixer is registered in the postblock
    assert any([isinstance(module, tracer_fixer) for module in postblock.modules()])

    # that tracer_fixer run
    input_dict = {'y_pred': input_tensor}
    output_dict = postblock(input_dict)

    # verify negative values
    assert output_dict['y_pred'].min() >= 0

def test_SKEBS_era5():
    """
    todo after implementation
    """
    pass

