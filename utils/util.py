import torch
import torch.nn as nn
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict
from model.decoding_model import DELTA
from model.pretrain_model import pretrainedmodule
from transformers import PreTrainedModel
from model.wrappers import Wrapper_pretrain
from typing import List

def get_linear_annealing_weight(step, start_step, end_step):
    """
    선형 스케줄에 따른 KL 가중치를 계산합니다.
    특정 스텝 구간(start_step ~ end_step) 동안 가중치가 0에서 1로 선형적으로 증가합니다.

    Args:
        step (int): 현재 학습 스텝.
        start_step (int): 어닐링이 시작되는 스텝 (가중치 0).
        end_step (int): 어닐링이 종료되는 스텝 (가중치 1).

    Returns:
        float: 계산된 KL 가중치 (0.0에서 1.0 사이).
    """
    if step < start_step:
        return 0.0
    # (step - start_step) / (end_step - start_step)을 계산하여 선형적으로 증가
    # min(..., 1.0)을 통해 가중치가 1을 넘지 않도록 보장
    return min(1.0, (step - start_step) / (end_step - start_step))


@dataclass
class BuildOptions:
    model_name: str = 'DELTA'  # 'DELTA', 'w/o_diffusion', 'DConv', 'EEGNet', 'w/o_pretrained'
    is_con: bool = False
    is_geo: bool = False
    is_kl: bool  = False
    is_diffusion: List[bool] = field(default_factory=lambda: [True, False])
    autoencoder: nn.Module = None
    denoising_module: nn.Module = None
    pretrained_module: Wrapper_pretrain = None
    pretrained_LM: PreTrainedModel = None
    device: torch.device = torch.device('cpu')
    noise_scheduler:str = 'linear'
    time_step:int = 1000

        
    def __call__(self):
        if 'pretrain' in self.model_name and 'pretrained' not in self.model_name:
            return self.autoencoder, self.denoising_module, self.pretrained_LM, self.is_con, self.is_geo, self.is_kl, self.is_diffusion, self.device, self.noise_scheduler, self.time_step
        else:
            return self.pretrained_module, self.is_diffusion, self.noise_scheduler, self.time_step, self.device


def build_delta(apts: BuildOptions):
    return DELTA(*apts()).to(apts.device)



def build_pretrain_module(apts: BuildOptions):
    return pretrainedmodule(*apts()).to(apts.device)


MODEL_BUILDERS: Dict[str, Callable[[], object]] = {
    "DELTA":           build_delta,
    "full_diffusion": build_delta,
    "main_diffusion": build_delta,
    "wo_diffusion":   build_delta,
    "DConv":           build_delta,
    "EEGNet":          build_delta,
    "wo_pretrained":  build_delta,
    "DConv_pretrain":           build_pretrain_module,
    "EEGNet_pretrain":          build_pretrain_module,
    "pretrain": build_pretrain_module,
    "wo_diffusion_pretrain": build_pretrain_module,
    }

def make_model(opts: BuildOptions):
    return MODEL_BUILDERS[opts.model_name](opts)  # 본체 생성





def set_seed(seed_val):
# random.seed(seed_val)
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        """
        Args:
            patience (int): 개선이 없다고 판단할 연속 epoch 수
            delta (float): '개선'으로 간주할 최소 손실 감소 폭
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss                       # 손실이 작을수록 좋은 모델
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads