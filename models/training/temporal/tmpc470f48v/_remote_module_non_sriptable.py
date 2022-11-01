from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef
from typing import Tuple  # pyre-ignore: unused import


module_interface_cls = None


def forward_async(self, *args, **kwargs):
    args = (self.module_rref, self.device, self.is_device_map_set, *args)
    kwargs = {**kwargs}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )


def forward(self, *args, **kwargs):
    args = (self.module_rref, self.device, self.is_device_map_set, *args)
    kwargs = {**kwargs}
    ret_fut = rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )
    return ret_fut.wait()


_generated_methods = [
    forward_async,
    forward,
]




def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, *args, **kwargs):
    module = module_rref.local_value()
    device = torch.device(device)

    if device.type != "cuda":
        return module.forward(*args, **kwargs)

    # If the module is on a cuda device,
    # move any CPU tensor in args or kwargs to the same cuda device.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.to(device) if isinstance(i, Tensor) else i for i in *args)``.
    args = (*args,)
    out_args: Tuple[()] = ()
    for arg in args:
        arg = (arg.to(device),) if isinstance(arg, Tensor) else (arg,)
        out_args = out_args + arg

    kwargs = {**kwargs}
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = kwargs[k].to(device)

    if is_device_map_set:
        return module.forward(*out_args, **kwargs)

    # If the device map is empty, then only CPU tensors are allowed to send over wire,
    # so have to move any GPU tensor to CPU in the output.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.cpu() if isinstance(i, Tensor) else i for i in module.forward(*out_args, **kwargs))``.
    ret: Tuple[()] = ()
    for i in module.forward(*out_args, **kwargs):
        i = (i.cpu(),) if isinstance(i, Tensor) else (i,)
        ret = ret + i
    return ret
