def wrap_non_distributed_model(model, device='cuda', dim=0, *args, **kwargs):

    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model.npu(), dim=dim, *args, **kwargs)
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        model = MLUDataParallel(model.mlu(), dim=dim, *args, **kwargs)
    elif device == 'cuda':
        from mmcv.parallel import MMDataParallel
        model = MMDataParallel(model.cuda(), dim=dim, *args, **kwargs)
    elif device == 'cpu':
        model = model.cpu()
    elif device == 'ipu':
        model = model.cpu()
    elif device == 'mps':
        from mmcv.device import mps
        model = mps.MPSDataParallel(model.to('mps'), dim=dim, *args, **kwargs)
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model


def wrap_distributed_model(model, device='cuda', *args, **kwargs):
    if device == 'npu':
        from mmcv.device.npu import NPUDistributedDataParallel
        from torch.npu import current_device
        model = NPUDistributedDataParallel(
            model.npu(), *args, device_ids=[current_device()], **kwargs)
    elif device == 'mlu':
        import os

        from mmcv.device.mlu import MLUDistributedDataParallel
        model = MLUDistributedDataParallel(
            model.mlu(),
            *args,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            **kwargs)
    elif device == 'cuda':
        from mmcv.parallel import MMDistributedDataParallel
        from torch.cuda import current_device
        model = MMDistributedDataParallel(
            model.cuda(), *args, device_ids=[current_device()], **kwargs)
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model
