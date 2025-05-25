import math
from typing import Tuple

import deep_gemm
import torch
import torch.nn.functional as F
from deep_gemm import ceil_div, get_col_major_tma_aligned_tensor
from torch.nn import functional as F
from torch.nn import init


def pad_to_128(tensor):
    """
    """
    H, W = tensor.shape
    pad_amount = (-H) % 128   
    
    if pad_amount == 0:
        return tensor  
     
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_amount))  
    
    return padded_tensor


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


 
def dw_grad(grad_output, input):

    device = torch.cuda.current_device()
    grad_output = grad_output.contiguous()
    input = input.contiguous()

    grad_output_view = grad_output.view(-1, grad_output.shape[-1])
    input_view = input.view(-1, input.shape[-1])

    grad_output_view = pad_to_128(grad_output_view)
    input_view = pad_to_128(input_view)

    in_feature = input_view.shape[-1]
    out_feature = grad_output_view.shape[-1]

    grad_output_view_t = grad_output_view.t().contiguous()
    input_view_t = input_view.t().contiguous()
    out = torch.zeros((out_feature, in_feature), device=device, dtype=torch.float)  
    # ref_out =   (grad_output_view_t.float() @ input_view_t.float().t())
    x_fp8 = per_token_cast_to_fp8(grad_output_view_t)
    y_fp8 = per_token_cast_to_fp8(input_view_t)
    # NOTES: please do inplace add on the `out` later
    deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt(x_fp8, y_fp8, out)
    return out

"""
grad_output = torch.ones((bs, seq_len, out_feature), device='cuda', dtype=torch.bfloat16)
input = torch.randn((bs, seq_len, in_feature), device='cuda', dtype=torch.bfloat16)
dw_grad(grad_output, input)
"""

 
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        bs = input.shape[0]
        input_view = input.view(-1, input.shape[-1])
        m = input_view.shape[0]
        n = weight.shape[0]
        x_fp8, y_fp8 = per_token_cast_to_fp8(input_view), per_block_cast_to_fp8(weight)
        # Transpose earlier so that the testing will not trigger transposing kernels
        x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
        output = torch.zeros((m, n), device=torch.cuda.current_device(), dtype=torch.bfloat16)
        deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, output)
        output = output.view(bs, -1, n)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)  # 添加偏置
        # print("fp is called")
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # 计算输入和权重的梯度
        grad_output = grad_output.contiguous()
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)  # 输入梯度
            bs = grad_output.shape[0]
            grad_output_view = grad_output.view(-1, grad_output.shape[-1])
            m = grad_output_view.shape[0]
            n = weight.shape[1]
            y_fp8 = per_block_cast_to_fp8(weight.t())
            x_fp8 = per_token_cast_to_fp8(grad_output_view) 
            grad_input = torch.zeros((m, n), device=torch.cuda.current_device(), dtype=torch.bfloat16)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, grad_input)
            grad_input = grad_input.view(bs, -1, n).bfloat16()

        if ctx.needs_input_grad[1]: 
            grad_weight = dw_grad(grad_output, input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)  
        return grad_input, grad_weight, grad_bias

 
class FP8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)