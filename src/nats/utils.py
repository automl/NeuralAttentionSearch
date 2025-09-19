import torch
import inspect

def get_kwargs():
    # https://stackoverflow.com/a/65927265
    # get the values and arguments of the current function
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs


def check_fp16_dtype():
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] >= 8:
        fp16_dtype = 'bfloat16'
    else:
        fp16_dtype = 'float16'
    return fp16_dtype

