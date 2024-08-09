'''
parser.py
-------------------------------------------------------
Content:
    - CREDIT_main_parser

Yingkai Sha
ksha@ucar.edu
'''

import os
import warnings

import numpy as np
import xarray as xr

def CREDIT_main_parser(conf, parse_training=True, parse_predict=True, print_summary=False):
    '''
    This function examines the config.yml input, and produce its standardized version.
    Missing keywords will either trigger assertion errors or receive a defualt value.
    All other components of this repo relies on CREDIT_main_parser.
    ----------------------------------------------------------------------------------
    Where is it applied?
        - applications/train.py
        - applications/train_multistep.py
        - applications/rollout_to_netcdf_new.py
    '''
    
    assert 'save_loc' in conf, "save location of the CREDIT project ('save_loc') is missing from conf"
    assert 'data' in conf, "data section ('data') is missing from conf"
    assert 'model' in conf, "model section ('model') is missing from conf"
    assert 'latitude_weights' in conf['loss'], (
        "lat / lon file ('latitude_weights') is missing from conf['loss']")
    
    if parse_training:
        assert 'trainer' in conf, "trainer section ('trainer') is missing from conf"
        assert 'loss' in conf, "loss section ('loss') is missing from conf"
    
    if parse_predict:
        assert 'predict' in conf, "predict section ('predict') is missing from conf"
    
    conf['save_loc'] = os.path.expandvars(conf['save_loc'])
    
    # --------------------------------------------------------- #
    # conf['data'] section

    # must have upper-air variables
    assert 'variables' in conf['data'], "upper-air variable names ('variables') is missing from conf['data']"
    
    if (conf['data']['variables'] is None) or (len(conf['data']['variables']) == 0):
        print("Upper-air variable name conf['data']['variables']: {} cannot be processed".format(conf['data']['variables']))
        raise
        
    assert 'save_loc' in conf['data'], "upper-air var save locations ('save_loc') is missing from conf['data']"
    
    if conf['data']['save_loc'] is None:
        print("Upper-air var save locations conf['data']['save_loc']: {} cannot be processed".format(conf['data']['save_loc']))
        raise

    if 'levels' not in conf['data']: 
        if 'levels' in conf['model']:
            conf['data']['levels'] = conf['model']['levels']
        else:
            print("number of upper-air levels ('levels') is missing from both conf['data'] and conf['model']")
            raise
    
    # surface inputs
    if 'surface_variables' in conf['data']:
        if conf['data']['surface_variables'] is None:
            conf['data']['flag_surface'] = False
        elif len(conf['data']['surface_variables']) > 0:
            conf['data']['flag_surface'] = True
            assert 'save_loc_surface' in conf['data'], (
                "surface var save locations ('save_loc_surface') is missing from conf['data']")
        else:
            conf['data']['flag_surface'] = False
    else:
        conf['data']['flag_surface'] = False
    
    # dyn forcing inputs
    if 'dynamic_forcing_variables' in conf['data']:
        if conf['data']['dynamic_forcing_variables'] is None:
            conf['data']['flag_dyn_forcing'] = False
        elif len(conf['data']['dynamic_forcing_variables']) > 0:
            conf['data']['flag_dyn_forcing'] = True
            assert 'save_loc_dynamic_forcing' in conf['data'], (
                "dynamic forcing var save locations ('save_loc_dynamic_forcing') is missing from conf['data']")
        else:
            conf['data']['flag_dyn_forcing'] = False
    else:
        conf['data']['flag_dyn_forcing'] = False
    
    # diagnostic outputs
    if parse_training:
        if 'diagnostic_variables' in conf['data']:
            if conf['data']['diagnostic_variables'] is None:
                conf['data']['flag_diagnostic'] = False
            elif len(conf['data']['diagnostic_variables']) > 0:
                conf['data']['flag_diagnostic'] = True
                assert 'save_loc_diagnostic' in conf['data'], (
                    "diagnostic var save locations ('save_loc_diagnostic') is missing from conf['data']")
            else:
                conf['data']['flag_diagnostic'] = False
        else:
            conf['data']['flag_diagnostic'] = False
    
    # forcing inputs
    if 'forcing_variables' in conf['data']:
        if conf['data']['forcing_variables'] is None:
            conf['data']['flag_forcing'] = False
        elif len(conf['data']['forcing_variables']) > 0:
            conf['data']['flag_forcing'] = True
            assert 'save_loc_forcing' in conf['data'], (
                "forcing var save locations ('save_loc_forcing') is missing from conf['data']")
        else:
            conf['data']['flag_forcing'] = False
    else:
        conf['data']['flag_forcing'] = False
        
    # static inputs
    if 'static_variables' in conf['data']:
        if conf['data']['static_variables'] is None:
            conf['data']['flag_static'] = False
        elif len(conf['data']['static_variables']) > 0:
            conf['data']['flag_static'] = True
            assert 'save_loc_static' in conf['data'], (
                "static var save locations ('save_loc_static') is missing from conf['data']")
        else:
            conf['data']['flag_static'] = False
    else:
        conf['data']['flag_static'] = False

    if conf['data']['flag_surface'] is False:
        conf['data']['save_loc_surface'] = None
        conf['data']['surface_variables'] = None

    if conf['data']['flag_dyn_forcing'] is False:
        conf['data']['save_loc_dynamic_forcing'] = None
        conf['data']['dynamic_forcing_variables'] = None

    if conf['data']['flag_diagnostic'] is False:
        conf['data']['save_loc_diagnostic'] = None
        conf['data']['diagnostic_variables'] = None

    if conf['data']['flag_forcing'] is False:
        conf['data']['save_loc_forcing'] = None
        conf['data']['forcing_variables'] = None
    
    if conf['data']['flag_static'] is False:
        conf['data']['save_loc_static'] = None
        conf['data']['static_variables'] = None
    
    ## I/O data sizes
    if parse_training:
        assert 'train_years' in conf['data'], (
            "year range for training ('train_years') is missing from conf['data']")

        # 'valid_years' is required even for conf['trainer']['skip_validation']: True
        # 'valid_years' and 'train_years' can overlap
        assert 'valid_years' in conf['data'], (
            "year range for validation ('valid_years') is missing from conf['data']")
        
        assert 'forecast_len' in conf['data'], (
            "Number of time frames for loss compute ('forecast_len') is missing from conf['data']")
    
        if 'valid_history_len' not in conf['data']:
            # use "history_len" for "valid_history_len"
            conf['data']['valid_history_len'] = conf['data']['history_len']
        
        if 'valid_forecast_len' not in conf['data']:
            # use "forecast_len" for "valid_forecast_len"
            conf['data']['valid_forecast_len'] = conf['data']['forecast_len']
            
        if 'max_forecast_len' not in conf['data']:
            conf['data']['max_forecast_len'] = None #conf['data']['forecast_len']
        
        # one_shot
        if 'one_shot' not in conf['data']:
            conf['data']['one_shot'] = None
    
    assert 'history_len' in conf['data'], "Number of input time frames ('history_len') is missing from conf['data']"
    assert 'lead_time_periods' in conf['data'], "Number of forecast hours ('lead_time_periods') is missing from conf['data']"
    assert 'scaler_type' in conf['data'], "'scaler_type' is missing from conf['data']"
    
    if conf['data']['scaler_type'] == 'std_new':
        assert 'mean_path' in conf['data'], "The z-score mean file ('mean_path') is missing from conf['data']"
        assert 'std_path' in conf['data'], "The z-score std file ('std_path') is missing from conf['data']"
        
    # skip_periods
    if 'skip_periods' not in conf['data']:
        conf['data']['skip_periods'] = None
        
    if 'static_first' not in conf['data']:
        conf['data']['static_first'] = True
    
    # --------------------------------------------------------- #
    # conf['trainer'] section
    
    if parse_training:
        
        assert 'mode' in conf['trainer'], "Resource type ('mode') is missing from conf['trainer']"
        assert 'type' in conf['trainer'], "Training strategy ('type') is missing from conf['trainer']"
        
        assert 'load_weights' in conf['trainer'], "must specify 'load_weights' in conf['trainer']"
        assert 'learning_rate' in conf['trainer'], "must specify 'learning_rate' in conf['trainer']"
        
        assert 'batches_per_epoch'  in conf['trainer'], (
            "Number of training batches per epoch ('batches_per_epoch') is missing from onf['trainer']")
        
        assert 'train_batch_size'  in conf['trainer'], (
            "Training set batch size ('train_batch_size') is missing from onf['trainer']")
    
        if 'thread_workers' not in conf['trainer']:
            conf['trainer']['thread_workers'] = 4
        
        if 'skip_validation' not in conf['trainer']:
            conf['trainer']['skip_validation'] = False
    
        if conf['trainer']['skip_validation'] is False:
    
            if 'valid_thread_workers' not in conf['trainer']:
                conf['trainer']['valid_thread_workers'] = 0
            
            assert 'valid_batch_size'  in conf['trainer'], (
                "Validation set batch size ('valid_batch_size') is missing from onf['trainer']")
            
            assert 'valid_batches_per_epoch'  in conf['trainer'], (
                "Number of validation batches per epoch ('valid_batches_per_epoch') is missing from onf['trainer']")
            
        if 'use_scheduler' in conf['trainer']:
            # lr will be controlled by scheduler
            conf['trainer']['update_learning_rate'] = False
            
            assert 'scheduler' in conf['trainer'], (
                "must specify 'scheduler' in conf['trainer'] if a scheduler is used")
            
            assert 'load_optimizer' in conf['trainer'], (
                "must specify 'load_optimizer' in conf['trainer'] if a scheduler is used")
            
            assert 'reload_epoch' in conf['trainer'], (
                "must specify 'reload_epoch' in conf['trainer'] if a scheduler is used")
        
        if 'update_learning_rate' not in conf['trainer']:
            conf['trainer']['update_learning_rate'] = False
            
        if 'train_one_epoch' not in conf['trainer']:
            conf['trainer']['train_one_epoch'] = False
    
        if conf['trainer']['train_one_epoch'] is False:
            assert 'start_epoch' in conf['trainer'], "must specify 'start_epoch' in conf['trainer']"
            assert 'epochs' in conf['trainer'], "must specify 'epochs' in conf['trainer']"
        else:
            conf['trainer']['epochs'] = 999
            if 'num_epoch' in conf['trainer']:
                warnings.warn(
                    "conf['trainer']['num_epoch'] will be overridden by conf['trainer']['train_one_epoch']: True")
            
        if 'amp' not in conf['trainer']:
            conf['trainer']['amp'] = False
            
        if 'weight_decay' not in conf['trainer']:
            conf['trainer']['weight_decay'] = 0
            
        if 'stopping_patience' not in conf['trainer']:
            conf['trainer']['stopping_patience'] = 999
    
        if 'activation_checkpoint' not in conf['trainer']:
            conf['trainer']['activation_checkpoint'] = True
    
        if 'cpu_offload' not in conf['trainer']:
            conf['trainer']['cpu_offload'] = False
    
        if 'grad_accum_every' not in conf['trainer']:
            conf['trainer']['grad_accum_every'] = 1
    
        if 'grad_max_norm' not in conf['trainer']:
            conf['trainer']['grad_max_norm'] = 1.0
    
    # --------------------------------------------------------- #
    # conf['loss'] section
    
    if parse_training:
        assert 'training_loss' in conf['loss'], "Training loss ('training_loss') is missing from conf['loss']"
        assert 'use_latitude_weights' in conf['loss'], "must specify 'use_latitude_weights' in conf['loss']"
        assert 'use_variable_weights' in conf['loss'], "must specify 'use_variable_weights' in conf['loss']"
    
        if conf['loss']['use_variable_weights']:
            assert 'variable_weights' in conf['loss'], (
                "must specify 'variable_weights' in conf['loss'] if 'use_variable_weights': True")
    
        if 'use_power_loss' not in conf['loss']:
            conf['loss']['use_power_loss'] = False
    
        if 'use_spectral_loss' not in conf['loss']:
            conf['loss']['use_spectral_loss'] = False
            
        if conf['loss']['use_power_loss'] and conf['loss']['use_spectral_loss']:
            warnings.warn(
                "'use_power_loss: True' and 'use_spectral_loss: True' are both applied")
            
        if conf['loss']['use_power_loss'] or conf['loss']['use_spectral_loss']:
            if 'spectral_lambda_reg' not in conf['loss']:
                conf['loss']['spectral_lambda_reg'] = 0.1
    
            if 'spectral_wavenum_init' not in conf['loss']:
                conf['loss']['spectral_wavenum_init'] = 20
                
    # --------------------------------------------------------- #
    # conf['parse_predict'] section
    
    if parse_predict:
        assert 'forecasts' in conf['predict'], "Rollout settings ('forecasts') is missing from conf['predict']"
        assert 'save_forecast' in conf['predict'], (
            "Rollout save location ('save_forecast') is missing from conf['predict']")
        
        if 'use_laplace_filter' not in conf['predict']:
            conf['predict']['use_laplace_filter'] = False
            
        if 'metadata' not in conf['predict']:
            conf['predict']['metadata'] = False
    
        if 'save_vars' not in conf['predict']:
            conf['predict']['save_vars'] = []
    
        if 'mode' not in conf['predict']:
            if 'mode' in conf['trainer']:
                conf['predict']['mode'] = conf['trainer']['mode']
            else:
                print("Resource type ('mode') is missing from both conf['trainer'] and conf['predict']")
                raise

    # ==================================================== #
    # print summary
    if print_summary:
        print('Upper-air variables: {}'.format(conf['data']['variables']))
        print('Surface variables: {}'.format(conf['data']['surface_variables']))
        print('Dynamic forcing variables: {}'.format(conf['data']['dynamic_forcing_variables']))
        print('Diagnostic variables: {}'.format(conf['data']['diagnostic_variables']))
        print('Forcing variables: {}'.format(conf['data']['forcing_variables']))
        print('Static variables: {}'.format(conf['data']['static_variables']))
        
    return conf




