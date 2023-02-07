def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'grconvnet_origin':
        from .grconvnet_origin import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet_most':
        from .grconvnet_most import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet_mish_allpw':
        from .grconvnet_mish_allpw import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet_mish_2pw':
        from .grconvnet_mish_2pw import GenerativeResnet
        return GenerativeResnet
    
    elif network_name == 'grconvnet_mbc':
        from .grconvnet_mbc import GenerativeResnet
        return GenerativeResnet

    
    elif network_name == 'grconvnet_fuse':
        from .grconvnet_mbc import GenerativeResnet
        return GenerativeResnet
    
    elif network_name == 'grconvnet_graduate':
        from .grconvnet_graduate import GenerativeResnet
        return GenerativeResnet

    elif network_name == 'grconvnet_graduate2':
        from .grconvnet_graduate2 import GenerativeResnet
        return GenerativeResnet
    # elif network_name == 'mbc_dropout':
    #     from .mbc_dropout import GenerativeResnet
    #     return GenerativeResnet
        
    elif network_name == 'resdense_5':
        from .resdense_5 import GenerativeResnet
        return GenerativeResnet
    
    elif network_name == 'grconv_mish_se':
        from .grconv_mish_se import GenerativeResnet
        return GenerativeResnet

    elif network_name == 'graduation':
        from .graduation import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'graduation2':
        from .graduation2 import GenerativeResnet
        return GenerativeResnet
    
    elif network_name == 'graduation3':
        from .graduation3 import GenerativeResnet
        return GenerativeResnet
    

    else:
        
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
