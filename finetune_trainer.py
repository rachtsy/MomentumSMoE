import os, sys
import argparse
import math, random
import torch
import tqdm
import pdb
import torch.nn as nn
from custom_gates import *


def _train_step(model, load_balance, X, Y, h_cache, eval_only, loss_div=1):
    """Single training step."""
    acc_num, acc_value = 0, 0.0
    out, h_cache = model(X, h_cache)
    out = out.view(-1, out.size(-1))
    # loss = torch.nn.functional.nll_loss(out, Y.view(-1))
    criterion = nn.CrossEntropyLoss()
    # pdb.set_trace()
    loss = criterion(out, Y)
    loss_value = loss.item() / loss_div
    # loss = loss.float()
    # loss = loss.item() / loss_div
    acc_value += (out.argmax(-1) == Y).sum().item()
    acc_num += Y.shape[0]

    if not eval_only:
        # loss term from adaptive-span
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            loss += sum(
                model.module.layers[layer_i].attn.attn.adaptive_span.get_loss()
                for layer_i in range(model.module.attn_layer_count)
            )

        if load_balance > 0:
            balance_loss = 0
            for name, m in model.named_modules():
                if isinstance(m, CustomNaiveGate_Balance_SMoE) or isinstance(
                    m, CustomNaiveGate_Balance_XMoE
                ):
                    balance_loss += m.loss
            loss += load_balance * balance_loss

        (loss / loss_div).backward()
    return loss_value, acc_num, acc_value, h_cache


def _train_batch(
    model, load_balance, optimizer, scheduler, X, Y, h_cache, eval_only, batch_split
):
    """Train on a batch."""

    optimizer.zero_grad()
    total_len, total_acc = 0, 0.0
    if batch_split == 1:
        # process a batch in a single step (default behaviour)
        loss_value, total_len, total_acc, h_cache = _train_step(
            model, load_balance, X, Y, h_cache, eval_only
        )
    else:
        # split a batch into multiple pieces that each can fit in memory
        assert X.size(0) % batch_split == 0
        split_size = X.size(0) // batch_split
        loss_value = 0
        h_cache_list = []
        for split_ind in range(batch_split):
            split_slice = slice(split_ind * split_size, (split_ind + 1) * split_size)
            split_h_cache = [h[split_slice, :, :] for h in h_cache]
            tmp_len, tmp_acc = 0, 0.0
            # pdb.set_trace()
            split_loss_value, tmp_len, tmp_acc, split_h_cache = _train_step(
                model=model,
                load_balance=load_balance,
                X=X[split_slice, :],
                Y=Y[split_slice],
                h_cache=split_h_cache,
                eval_only=eval_only,
                loss_div=batch_split,
            )
            loss_value += split_loss_value
            total_len += tmp_len
            total_acc += tmp_acc
            h_cache_list.append(split_h_cache)
        h_cache = [
            torch.cat([h_cache_list[i][l] for i in range(batch_split)], dim=0)
            for l in range(len(h_cache))
        ]
    if not eval_only:
        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # make sure span parameters are in a correct range
        if model.module.layers[0].attn.attn.adapt_span_enabled:
            for layer in model.module.layers:
                if layer.use_attn:
                    layer.attn.attn.adaptive_span.clamp_param()
    return loss_value, total_len, total_acc, h_cache


def train_iteration(
    model,
    load_balance,
    optimizer,
    scheduler,
    data,
    nb_batches_per_iter,
    block_size,
    eval_only,
    train_pos,
    h_cache,
    batch_split,
    checkpoint_path,
):
    """Single training iteration."""
    if eval_only:
        model.eval()
    else:
        model.train()

    nb_batches_per_iter_max = nb_batches_per_iter
    if eval_only:
        # eval on fewer batches during training for speed-up
        nb_batches_per_iter_max = max(1, nb_batches_per_iter // 10)
        for _temp, _, _ in data:
            ch_block = _temp.size(1)  # data.size(1)
            break
        nb_batches_per_iter_max = min(
            nb_batches_per_iter_max, math.ceil(ch_block / block_size)
        )

    loss_all = 0
    # all accuracy and number
    total_len, total_acc = 0, 0.0
    actual_nb_batches_per_iter = 0
    for _data, _att_mask, _target in tqdm.tqdm(data):
        actual_nb_batches_per_iter += 1
        _data = _data.cuda()
        _target = _target.cuda()
        X = _data.permute(
            1, 0
        ).contiguous()  # data[:, train_pos: train_pos + block_size].contiguous()
        Y = _target  # .permute(1,0) #.contiguous() #data[:, train_pos + 1: train_pos + block_size + 1].contiguous()

        loss, tmp_len, tmp_acc, h_cache = _train_batch(
            model=model,
            load_balance=load_balance,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache,
            eval_only=eval_only,
            batch_split=batch_split,
        )
        loss_all += loss
        total_len += tmp_len
        total_acc += tmp_acc
        train_pos += block_size
        if train_pos >= _data.size(1) - block_size:
            # reached the end. randomize the offset to reduce overfitting
            train_pos = random.randrange(block_size)
            # reset the cache
            for h in h_cache:
                h.fill_(0)

    loss_all = loss_all / actual_nb_batches_per_iter
    acc_all = 100 * total_acc / total_len
    return loss_all, acc_all, train_pos, h_cache


# do full evaluation
def full_eval(model, optimizer, scheduler, data, block_size, hidden_size):
    model.eval()
    train_pos = 0
    # nb_batches_per_iter_max = math.ceil(data.encoded.size(1) / block_size)
    h_cache = [
        torch.zeros(
            data.bsz,
            model.module.layers[layer_i].attn.attn.get_cache_size(),
            hidden_size,
        ).cuda()
        for layer_i in range(model.module.attn_layer_count)
    ]

    loss_all = 0
    actual_nb_batches_per_iter = 0
    total_len, total_acc = 0, 0.0

    for _data, _att_mask, _target in tqdm.tqdm(data):
        actual_nb_batches_per_iter += 1
        _data = _data.cuda()
        _target = _target.cuda()
        X = _data.permute(
            1, 0
        ).contiguous()  # data[:, train_pos: train_pos + block_size].contiguous()
        Y = _target  # .permute(1,0) #.contiguous() #data[:, train_pos + 1: train_pos + block_size + 1].contiguous()
        # pdb.set_trace()
        # for _ in tqdm.tqdm(range(nb_batches_per_iter_max)):
        #     actual_nb_batches_per_iter += 1
        #     X = data[:, train_pos: train_pos + block_size].contiguous()
        #     Y = data[:, train_pos + 1: train_pos + block_size + 1].contiguous()

        loss, tmp_len, tmp_acc, h_cache = _train_batch(
            model=model,
            load_balance=0,
            optimizer=optimizer,
            scheduler=scheduler,
            X=X,
            Y=Y,
            h_cache=h_cache,
            eval_only=True,
            batch_split=1,
        )
        loss_all += loss
        total_len += tmp_len
        total_acc += tmp_acc
        train_pos += block_size
        if train_pos >= _data.size(1) - block_size:
            # Skip the remaining tokens as it can't make a whole block.
            # An effect on performance should be negligable for a large data.
            break

    loss_all = loss_all / actual_nb_batches_per_iter
    acc_all = 100 * total_acc / total_len
    return loss_all, acc_all
