r"""
FMoE core layer
"""

import tree
import os
import torch
import torch.nn as nn

from custom_functions import prepare_forward, ensure_comm
from custom_functions import MOEScatter, MOEGather
from custom_functions import AllGather, Slice
from gates import NaiveGate

from fastermoe.config import switch_from_env
import random
import torch.nn.functional as F
from torchmetrics.regression import KLDivergence


def kl_divergence(softmax_1, softmax_2):
    kl_divergence = KLDivergence(log_prob=False).cuda()
    return kl_divergence(softmax_1, softmax_2)


def cal_mse_loss(input, target):
    mse_loss = nn.MSELoss()
    _loss = mse_loss(input, target)
    return _loss


def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(
    inp, gate, expert_fn, num_expert, world_size, **kwargs
):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode="floor"),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )

    x = tree.map_structure(scatter_func, inp)

    x = expert_fn(x, fwd_expert_count)

    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )

    outp = tree.map_structure(gather_func, x)
    return outp


fmoe_faster_schedule = False
if switch_from_env("FMOE_FASTER_SCHEDULE_ENABLE", False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoEOpt(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        slice_group=None,
        moe_group=None,
        moe_top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        freq=0.0,
        alpha=0.0,
        act_experts="shuffle",
        g_blance=False,
        opt_blance=False,
        combine_gate=False,
        opt_loss="mse",
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.freq = freq
        self.alpha = alpha
        self.act_experts = act_experts
        self.opt_blance = opt_blance
        self.combine_gate = combine_gate
        self.opt_loss = opt_loss
        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = moe_top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        self.gate = gate(d_model, num_expert, world_size, moe_top_k, g_blance)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def cal_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.num_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.num_expert
        return loss

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """

        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)
        flip_ = random.random()
        gate_top_k_idx, gate_score, gate_ = self.gate(moe_inp, return_all_scores=True)

        if self.training:
            if flip_ > (1 - self.freq):
                # all experts score
                gate_top_k_val_org, _ = torch.topk(
                    gate_, k=self.num_expert, dim=-1, largest=True, sorted=False
                )
                gate_top_k_val_org = gate_top_k_val_org.view(
                    -1, self.num_expert
                )  # (BxL) x 1 x top_k
                gate_score_org = F.softmax(gate_top_k_val_org, dim=-1)
                # activate all experts with shuffle index
                if self.act_experts == "shuffle":
                    # searching best routing optimal
                    gate_dense = torch.ones_like(
                        gate_
                    )  # average the importance of all experts
                    gate_top_k_val_opt, gate_top_k_idx_opt = torch.topk(
                        gate_dense,
                        k=self.num_expert,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )  # [.. x top_k]
                    gate_top_k_val_opt = gate_top_k_val_opt.view(
                        -1, self.num_expert
                    )  # (BxL) x 1 x num_experts
                    gate_score_opt = F.softmax(gate_top_k_val_opt, dim=-1)

                    if hasattr(self.gate, "dynamic_top_k"):
                        self.top_k = self.gate.dynamic_top_k

                    if self.gate_hook is not None:
                        self.gate_hook(gate_top_k_idx_opt, gate_score_opt, None)

                    # delete masked tensors
                    if self.mask is not None and self.mask_dict is not None:
                        # TODO: to fix
                        def delete_mask_func(tensor):
                            # to: (BxL') x d_model
                            tensor = tensor[mask == 0, :]
                            return tensor

                        mask = self.mask.view(-1)
                        moe_inp = tree.map_structure(delete_mask_func, moe_inp)
                        gate_top_k_idx_opt = gate_top_k_idx_opt[mask == 0, :]
                    bs = moe_inp.shape[0]
                    fwd_tmp = _fmoe_general_global_forward(
                        moe_inp,
                        gate_top_k_idx_opt,
                        self.expert_fn,
                        self.num_expert,
                        self.world_size,
                        experts=self.experts,
                    ).reshape(bs, self.num_expert, -1)
                    # cal norm of output experts
                    fwd_norm = torch.norm(fwd_tmp, dim=2)
                # activate all experts without shuffle index
                else:
                    # activate with grad
                    if self.opt_blance:
                        fwd_tmp = None
                        for i in range(self.num_expert):
                            
                            temp_ = (
                                moe_inp @ self.experts.htoh4.weight[i].T
                                + self.experts.htoh4.bias[i]
                            )
                            temp_ = F.relu(temp_)
                            temp_ = (
                                temp_ @ self.experts.h4toh.weight[i].T
                                + self.experts.h4toh.bias[i]
                            )
                            temp_ = torch.unsqueeze(temp_, -1)
                            if fwd_tmp is None:
                                fwd_tmp = temp_.clone()
                            else:
                                fwd_tmp = torch.concat([fwd_tmp, temp_], dim=-1)
                    # activate without grad
                    else:
                        fwd_tmp = None
                        for i in range(self.num_expert):
                            
                            temp_ = (
                                moe_inp @ self.experts.htoh4.weight[i].T
                                + self.experts.htoh4.bias[i]
                            )
                            temp_ = F.relu(temp_)
                            temp_ = (
                                temp_ @ self.experts.h4toh.weight[i].T
                                + self.experts.h4toh.bias[i]
                            )
                            temp_ = torch.unsqueeze(temp_, -1)
                            if fwd_tmp is None:
                                fwd_tmp = temp_.clone()
                            else:
                                fwd_tmp = torch.concat([fwd_tmp, temp_], dim=-1)
                    # cal norm of output experts
                    fwd_norm = torch.norm(fwd_tmp, dim=1)
                # ensemble with gate information
                if self.combine_gate:
                    fwd_norm = fwd_norm * 0.5 + gate_ * 0.5
                gate_top_k_val_optim, gate_top_k_idx_optim = torch.topk(
                    fwd_norm, k=self.top_k, dim=-1, largest=True, sorted=False
                )
                # if balance loss
                if self.opt_blance:
                    opt_bl_loss = self.cal_load_balance(fwd_norm, gate_top_k_idx_optim)
                # get output
                gate_top_k_val_optim = gate_top_k_val_optim.view(-1, self.top_k)
                # push low score to zeros
                gate_score2 = torch.zeros(
                    (gate_top_k_val_optim.shape[0], self.num_expert)
                ).cuda()
                gate_score2.fill_(-10e9)
                # fill with topk score value
                gate_score2 = gate_score2.scatter(
                    1, gate_top_k_idx_optim, gate_top_k_val_optim
                )
                gate_score_optimal = F.softmax(gate_score2, dim=1)
                # # calculate loss
                if self.opt_loss == "mse":
                    add_loss = cal_mse_loss(
                        gate_score_org,
                        gate_score_optimal,
                    )
                else:
                    add_loss = kl_divergence(
                        gate_score_org,
                        gate_score_optimal,
                    )
                # if balance loss
                if self.opt_blance:
                    add_loss += opt_bl_loss
                    # add to balance loss
                    self.gate.loss = add_loss * self.alpha
                else:
                    self.gate.loss = add_loss * self.alpha
                #update gate policy
                gate_top_k_idx = gate_top_k_idx_optim
                gate_score = F.softmax(gate_top_k_val_optim, dim=1)


        if hasattr(self.gate, "dynamic_top_k"):
            self.top_k = self.gate.dynamic_top_k

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(
            moe_inp,
            gate_top_k_idx,
            self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts,
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp
