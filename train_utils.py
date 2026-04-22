import torch
from torcheval.metrics.text import Perplexity
import random
from tqdm import tqdm
from data_utils import compute_normalized_token_entropy
import random
import csv
import numpy as np
import os
from transformers import get_cosine_schedule_with_warmup
perplexity_metric = Perplexity(ignore_index=-100)

def single_step_progressive_masking(harmony_tokens, total_stages, mask_token_id, stage_in=None, bar_token_id=None):
    device = harmony_tokens.device
    B, L = harmony_tokens.shape
    total_stages = L
    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)
    if bar_token_id is not None:
        bar_mask = harmony_tokens == bar_token_id
        visible_harmony[bar_mask] = bar_token_id
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    target_indices = torch.zeros((B,), device=device)
    for b in range(B):
        stage = stage_indices[b] if stage_in is None else stage_in
        percent_visible = stage / total_stages
        perm = torch.randperm(L, device=device)
        num_visible = int(L * percent_visible)
        num_predict = 1
        visible_idx = perm[:num_visible]
        predict_idx = perm[:num_visible + num_predict]
        target_indices[b] = predict_idx[-1]
        visible_harmony[b, visible_idx] = harmony_tokens[b, visible_idx]
        denoising_target[b, predict_idx] = harmony_tokens[b, predict_idx]
    return (visible_harmony, denoising_target, stage_indices, target_indices)

def random_progressive_masking(harmony_tokens, total_stages, mask_token_id, stage_in=None, bar_token_id=None):
    device = harmony_tokens.device
    B, L = harmony_tokens.shape
    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)
    if bar_token_id is not None:
        bar_mask = harmony_tokens == bar_token_id
        visible_harmony[bar_mask] = bar_token_id
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    for b in range(B):
        stage = stage_indices[b] if stage_in is None else stage_in
        percent_visible = stage / total_stages
        percent_predict = 1 / total_stages
        perm = torch.randperm(L, device=device)
        num_visible = int(L * percent_visible)
        num_predict = int(L * percent_predict + 0.5)
        visible_idx = perm[:num_visible]
        predict_idx = perm[num_visible:num_visible + num_predict]
        visible_harmony[b, visible_idx] = harmony_tokens[b, visible_idx]
        denoising_target[b, predict_idx] = harmony_tokens[b, predict_idx]
    return (visible_harmony, denoising_target, stage_indices)

def full_to_partial_masking(harmony_tokens, mask_token_id, num_visible=0, bar_token_id=None):
    device = harmony_tokens.device
    B, L = harmony_tokens.shape
    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)
    if bar_token_id is not None:
        bar_mask = harmony_tokens == bar_token_id
        visible_harmony[bar_mask] = bar_token_id
        denoising_target[bar_mask] = bar_token_id
    perm = torch.randperm(L, device=device)
    visible_idx = perm[:num_visible]
    predict_idx = perm[num_visible:]
    visible_harmony[:, visible_idx] = harmony_tokens[:, visible_idx]
    denoising_target[:, predict_idx] = harmony_tokens[:, predict_idx]
    return (visible_harmony, denoising_target)

def structured_progressive_masking(harmony_tokens, total_stages, mask_token_id, stage_in=None):
    B, L = harmony_tokens.shape
    device = harmony_tokens.device
    visible_harmony = torch.full_like(harmony_tokens, mask_token_id)
    denoising_target = harmony_tokens.clone()
    input_unmask = torch.full_like(harmony_tokens, 0, dtype=torch.bool, device=device)
    target_to_learn = torch.full_like(harmony_tokens, False, dtype=torch.bool, device=device)
    stage_indices = torch.randint(0, total_stages, (B,), device=device)
    for i in range(B):
        stage = stage_indices[i] if stage_in is None else stage_in
        spacing_target = min(L, max(1, int(2 ** (8 - stage))))
        target_to_learn[i, ::spacing_target] = True
        spacing_input = 2 * spacing_target
        input_unmask[i, ::spacing_input] = spacing_input <= L
    visible_harmony[input_unmask] = harmony_tokens[input_unmask]
    target_to_learn[input_unmask] = False
    denoising_target[~torch.logical_or(target_to_learn, input_unmask)] = -100
    return (visible_harmony, denoising_target, stage_indices)

def apply_masking(harmony_tokens, mask_token_id, total_stages=10, curriculum_type='random', stage=None, bar_token_id=None):
    if curriculum_type == 'random':
        return random_progressive_masking(harmony_tokens, total_stages, mask_token_id, stage, bar_token_id)
    elif curriculum_type == 'base2':
        return structured_progressive_masking(harmony_tokens, total_stages, mask_token_id, stage)
    elif curriculum_type == 'step':
        return single_step_progressive_masking(harmony_tokens, total_stages, mask_token_id, stage)

def apply_structured_masking(harmony_tokens, mask_token_id, stage, time_sigs, total_stages=10, curriculum_type='no'):
    B, T = harmony_tokens.shape
    masked_harmony = torch.full_like(harmony_tokens, mask_token_id)
    target = harmony_tokens.clone()
    device = harmony_tokens.device
    target_to_learn = torch.full_like(harmony_tokens, curriculum_type == 'random' or curriculum_type == 'no', dtype=torch.bool, device=device)
    if curriculum_type == 'ts_incr':
        input_unmask = torch.full_like(harmony_tokens, 0, dtype=torch.bool, device=device)
    if curriculum_type == 'ts_incr' or curriculum_type == 'ts_blank':
        for i in range(B):
            ts_num = torch.nonzero(time_sigs[i, :14])[0] + 2
            ts_den = torch.nonzero(time_sigs[i, 14:])[0] * 8 - 4
            spacing_target = min(128, max(1, int(2 ** (7 * stage / total_stages) * (ts_num / ts_den))))
            target_to_learn[i, ::spacing_target] = True
            if curriculum_type == 'ts_incr':
                spacing_input = max(2, int(2 ** (8 * stage / total_stages) * (ts_num / ts_den)))
                input_unmask[i, ::spacing_input] = True
    if curriculum_type == 'random':
        for i in range(B):
            stage_ratio = 1.0 - (stage + 1) / total_stages
            valid_indices = (harmony_tokens[i] != -1).nonzero(as_tuple=False).squeeze()
            n_reveal = int(len(valid_indices) * stage_ratio)
            reveal_indices = random.sample(valid_indices.tolist(), n_reveal)
            masked_harmony[i, reveal_indices] = harmony_tokens[i, reveal_indices]
            target_to_learn = masked_harmony == mask_token_id
    if curriculum_type == 'ts_incr':
        masked_harmony[input_unmask] = harmony_tokens[input_unmask]
        target_to_learn[input_unmask] = False
        target[~torch.logical_or(target_to_learn, input_unmask)] = -100
    if curriculum_type == 'ts_blank':
        target[~target_to_learn] = -100
    return (masked_harmony, target)

def get_stage_linear(epoch, epochs_per_stage, max_stage):
    return min(epoch // epochs_per_stage, max_stage)

def get_stage_mixed(epoch, max_epoch, max_stage):
    if epoch >= max_epoch - 1:
        return max_stage
    progress = epoch / max_epoch
    probs = torch.softmax(torch.tensor([(1.0 - abs(progress - i / max_stage)) * 5 for i in range(max_stage + 1)]), dim=0)
    return torch.multinomial(probs, 1).item()

def get_stage_uniform(epoch, max_epoch, max_stage):
    return np.random.randint(max_stage + 1)

def apply_focal_sharpness(melody_grid, target_indices, focal_sharpness, min_sigma=1.0, max_sigma=40.0):
    B, L, dm = melody_grid.shape
    sigma = max_sigma - focal_sharpness * (max_sigma - min_sigma)
    positions = torch.arange(L, device=melody_grid.device).unsqueeze(0).expand(B, L)
    if target_indices.dim() == 1:
        target_indices = target_indices.unsqueeze(-1)
    elif target_indices.dim() == 2 and target_indices.shape[0] == 1:
        target_indices = target_indices.t()
    focal = target_indices.expand(-1, L)
    dist2 = (positions - focal).float() ** 2
    weights = torch.exp(-dist2 / (2 * sigma ** 2))
    weights = weights / weights.max(dim=1, keepdim=True).values
    weights = weights.unsqueeze(-1)
    attenuated_grid = melody_grid * weights
    return attenuated_grid

def validation_loop(curriculum_type, model, valloader, mask_token_id, bar_token_id, num_visible, condition_dim, total_stages, loss_fn, epoch, step, train_loss, train_accuracy, train_perplexity, train_token_entropy, best_val_loss, saving_version, results_path=None, transformer_path=None, tqdm_position=0):
    device = model.device
    model.eval()
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        running_perplexity = 0
        val_perplexity = 0
        running_token_entropy = 0
        val_token_entropy = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                perplexity_metric.reset()
                melody_grid = batch['pianoroll'].to(device)
                harmony_gt = batch['harmony_ids'].to(device)
                if condition_dim is not None:
                    conditioning_vec = batch['time_signature'].to(device)
                else:
                    conditioning_vec = None
                if curriculum_type == 'f2f':
                    harmony_input, harmony_target = full_to_partial_masking(harmony_gt, mask_token_id, num_visible, bar_token_id=bar_token_id)
                    stage_indices = None
                else:
                    harmony_input, harmony_target, stage_indices = apply_masking(harmony_gt, mask_token_id, total_stages=total_stages, curriculum_type=curriculum_type)
                    num_visible = -1
                logits = model(melody_grid.to(device), harmony_input.to(device), conditioning_vec, stage_indices)
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))
                batch_num += 1
                running_loss += loss.item()
                val_loss = running_loss / batch_num
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item() / mask.sum().item()
                val_accuracy = running_accuracy / batch_num
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                val_perplexity = running_perplexity / batch_num
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                val_token_entropy = running_token_entropy / batch_num
                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
    if transformer_path is not None:
        if curriculum_type == 'f2f' or best_val_loss > val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_path)
        if curriculum_type == 'f2f' and num_visible in [0, 5, 15, 30, 31, 50, 51]:
            torch.save(model.state_dict(), transformer_path[:-3] + f'_nvis{num_visible}.pt')
    print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open(results_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, num_visible, train_loss, train_accuracy, train_perplexity, train_token_entropy, val_loss, val_accuracy, val_perplexity, val_token_entropy, saving_version])
    return (best_val_loss, saving_version)

def train_with_curriculum(model, optimizer, trainloader, valloader, loss_fn, mask_token_id, curriculum_type='random', epochs=100, condition_dim=None, exponent=5, total_stages=10, results_path=None, transformer_path=None, bar_token_id=None, validations_per_epoch=1, tqdm_position=0):
    device = model.device
    perplexity_metric.to(device)
    best_val_loss = np.inf
    saving_version = 0
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', 'n_vis', 'train_loss', 'train_acc', 'train_ppl', 'train_te', 'val_loss', 'val_acc', 'val_ppl', 'val_te', 'sav_version']
        with open(results_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(result_fields)
    total_steps = len(trainloader) * epochs
    step = 0
    for epoch in range(epochs):
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        running_perplexity = 0
        train_perplexity = 0
        running_token_entropy = 0
        train_token_entropy = 0
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step} | trn')
            for batch in tepoch:
                perplexity_metric.reset()
                model.train()
                melody_grid = batch['pianoroll'].to(device)
                harmony_gt = batch['harmony_ids'].to(device)
                if condition_dim is not None:
                    conditioning_vec = batch['time_signature'].to(device)
                else:
                    conditioning_vec = None
                if curriculum_type == 'f2f':
                    if exponent == -1:
                        percent_visible = 0.0
                    else:
                        percent_visible = min(1.0, (step + 1) / total_steps) ** exponent
                    L = harmony_gt.shape[1]
                    num_visible = min(int(L * percent_visible), L - 1)
                    harmony_input, harmony_target = full_to_partial_masking(harmony_gt, mask_token_id, num_visible, bar_token_id=bar_token_id)
                    stage_indices = None
                else:
                    harmony_input, harmony_target, stage_indices = apply_masking(harmony_gt, mask_token_id, total_stages=total_stages, curriculum_type=curriculum_type)
                    num_visible = -1
                logits = model(melody_grid.to(device), harmony_input.to(device), conditioning_vec, stage_indices)
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss / batch_num
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item() / max(1, mask.sum().item())
                train_accuracy = running_accuracy / batch_num
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                train_perplexity = running_perplexity / batch_num
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                train_token_entropy = running_token_entropy / batch_num
                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
                step += 1
                if step % (total_steps // (epochs * validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_loop(curriculum_type, model, valloader, mask_token_id, bar_token_id, num_visible, condition_dim, total_stages, loss_fn, epoch, step, train_loss, train_accuracy, train_perplexity, train_token_entropy, best_val_loss, saving_version, results_path=results_path, transformer_path=transformer_path, tqdm_position=tqdm_position)
