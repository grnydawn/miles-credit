import torch
from credit.postblock import PostBlock
from credit.postblock import SKEBS
from credit.postblock import tracer_fixer
from credit.postblock import global_mass_fixer

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
    '''
    This function provides a functionality test on 
    tracer_fixer at credit.postblock
    '''
    
    # initialize post_conf, turn-off other blocks
    conf = {"post_conf": {"skebs": {'activate': False}}}
    conf['post_conf']['global_mass_fixer'] = {'activate': False}

    # tracer fixer specs
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
    output_tensor = postblock(input_dict)

    # verify negative values
    assert output_tensor.min() >= 0

def test_global_mass_fixer_rand():
    '''
    This function provides a I/O size test on 
    global_mass_fixer at credit.postblock
    '''
    # initialize post_conf, turn-off other blocks
    conf = {'post_conf': {'skebs': {'activate': False}}}
    conf['post_conf']['tracer_fixer'] = {'activate': False}
    conf['post_conf']['global_energy_fixer'] = {'activate': False}
    
    # global mass fixer specs
    conf['post_conf']['global_mass_fixer'] = {
        'activate': True, 
        'denorm': False, 
        'midpoint': False,
        'simple_demo': True, 
        'fix_level_num': 3,
        'q_inds': [0, 1, 2, 3, 4, 5, 6],
        'precip_ind': 7,
        'evapor_ind': 8
    }
    
    # data specs
    conf['post_conf']['data'] = {'lead_time_periods': 6}
    
    # initialize postblock
    postblock = PostBlock(**conf)

    # verify that global_mass_fixer is registered in the postblock
    assert any([isinstance(module, global_mass_fixer) for module in postblock.modules()])
    
    # input tensor
    x = torch.randn((1, 7, 2, 10, 18))
    # output tensor
    y_pred = torch.randn((1, 9, 1, 10, 18))
    
    input_dict = {"y_pred": y_pred, "x": x}
    
    # corrected output
    y_pred_fix = postblock(input_dict)

    # verify `y_pred_fix` and `y_pred` has the same size
    assert y_pred_fix.shape == y_pred.shape


def test_global_energy_fixer_rand():
    '''
    This function provides a I/O size test on 
    global_energy_fixer at credit.postblock
    '''
    # turn-off other blocks
    conf = {'post_conf': {'skebs': {'activate': False}}}
    conf['post_conf']['tracer_fixer'] = {'activate': False}
    conf['post_conf']['global_mass_fixer'] = {'activate': False}
    
    # global energy fixer specs
    conf['post_conf']['global_energy_fixer'] = {
        'activate': True,
        'simple_demo': True,
        'denorm': False,
        'midpoint': False,
        'T_inds': [0, 1, 2, 3, 4, 5, 6],
        'q_inds': [0, 1, 2, 3, 4, 5, 6],
        'U_inds': [0, 1, 2, 3, 4, 5, 6],
        'V_inds': [0, 1, 2, 3, 4, 5, 6],
        'TOA_rad_inds': [7, 8],
        'surf_rad_inds': [7, 8],
        'surf_flux_inds': [7, 8]}
    
    conf['post_conf']['data'] = {'lead_time_periods': 6}
    
    # initialize postblock
    postblock = PostBlock(**conf)

    # verify that global_energy_fixer is registered in the postblock
    assert any([isinstance(module, global_energy_fixer) for module in postblock.modules()])
    
    # input tensor
    x = torch.randn((1, 7, 2, 10, 18))
    # output tensor
    y_pred = torch.randn((1, 9, 1, 10, 18))
    
    input_dict = {"y_pred": y_pred, "x": x}
    # corrected output
    y_pred_fix = postblock(input_dict)
    
    assert y_pred_fix.shape == y_pred.shape


def test_SKEBS_era5():
    """
    todo after implementation
    """
    pass

