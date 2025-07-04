import random
import time
from typing import Dict, Iterator, List, Optional

import datasets
import numpy as np
import torch

from preference_datasets import get_collate_fn, get_dataset, tokenize_batch_element
from utils import (
    TemporarilySeededRandom,
    concatenated_forward,
    dpo_loss,
    slice_and_move_batch_for_device,
)


def get_shuffle_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    sft_mode: bool = False,
    pretrain_fraction: float = 1.0,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        pretrain_fraction: Fraction of the dataset to use for pretraining.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """
    assert (
        n_epochs is not None or n_examples is not None
    ), "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in dataset.items():
                this_flat_data.append(
                    (
                        prompt,
                        data["responses"],
                        data["pairs"],
                        data["sft_target"],
                        truncation_mode,
                    )
                )
            if split == "train":
                split_idx = int(pretrain_fraction * len(this_flat_data))
                if sft_mode:
                    this_flat_data = this_flat_data[:split_idx]
                else:
                    this_flat_data = this_flat_data[split_idx:]
            flat_data.extend(this_flat_data)
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    is_train = split == "train"
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            print("permutation_seeds", permutation_seeds, type(permutation_seeds))
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(
                    prompt,
                    sft_target,
                    sft_target,
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length,
                )
                batch_element = {
                    k: v for k, v in batch_element.items() if "rejected" not in k
                }
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(
                                f"Finished generating {n_examples} examples on {split} split"
                            )
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(
                        prompt,
                        responses[p[0]],
                        responses[p[1]],
                        truncation_mode,
                        tokenizer,
                        max_length,
                        max_prompt_length,
                    )
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(
                                    f"FINISHED {n_examples} EXAMPLES on {split} split"
                                )
                            done = True
                        batch = []

                    if not is_train:
                        break
        if done:
            break

        epoch_idx += 1


def get_active_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    policy: Optional[torch.nn.Module] = None,
    ref_policy: Optional[torch.nn.Module] = None,
    selection_strategy: str = "sharpe",  # 'sharpe', 'wsharpe', 'apl', 'exp'
    selection_ratio: float = 3.0,
    beta: float = 0.1,
    active_minibatch_split: int = 10,
    pretrain_fraction: float = 0.0,
    **kwargs,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        policy: pointer to current model.
        ref_policy: pointer to reference model.
        selection_strategy: 'sharpe', 'wsharpe', 'exp' or 'apl'.
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        active_minibatch_split: Number of microbatches to split the active selection into. This is useful for gradient accumulation.
        pretrain_fraction: Fraction of the dataset to use for pretraining.
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """
    assert policy is not None, "need a model for the active iterator"

    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in dataset.items():
                this_flat_data.append(
                    (
                        prompt,
                        data["responses"],
                        data["pairs"],
                        data["sft_target"],
                        truncation_mode,
                    )
                )
            if split == "train":
                split_idx = int(pretrain_fraction * len(this_flat_data))
                this_flat_data = this_flat_data[split_idx:]
            flat_data.extend(this_flat_data)

    # should now have flat_data = [(prompt, responses, pairs, sft_target, truncation_mode), ...]

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            for p in pairs:
                if done:
                    break
                batch_element = tokenize_batch_element(
                    prompt,
                    responses[p[0]],
                    responses[p[1]],
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length,
                )
                batch.append(batch_element)
                example_idx += 1
                if len(batch) >= batch_size * selection_ratio:
                    collated_batch = collate_fn(batch)
                    if "sharpe" in selection_strategy:
                        selected_batch = select_sharpe_elements(
                            batch=collated_batch,
                            num_to_select=batch_size,
                            policy=policy,
                            ref_policy=ref_policy,
                            selection_strategy=selection_strategy,
                            beta=beta,
                            gradient_acc_steps=active_minibatch_split,
                        )

                    elif "exp" in selection_strategy:
                        selected_batch = select_Exp_elements(
                            batch=collated_batch,
                            num_to_select=batch_size,
                            policy=policy,
                            ref_policy=ref_policy,
                            beta=beta,
                            gradient_acc_steps=active_minibatch_split,
                        )
                    elif "apl" in selection_strategy:
                        selected_batch = select_apl_elements(
                            batch=collated_batch,
                            num_to_select=batch_size,
                            policy=policy,
                            ref_policy=ref_policy,
                            beta=beta,
                            gradient_acc_steps=active_minibatch_split,
                        )
                    else:
                        raise NotImplementedError(
                            f"Selection strategy {selection_strategy} not implemented"
                        )
                    yield selected_batch
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f"FINISHED {n_examples} EXAMPLES on {split} split")
                        done = True
                    batch = []
        if done:
            break

        epoch_idx += 1


def select_sharpe_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    beta: float = 0.1,
    selection_strategy="sharpe",
    gradient_acc_steps=12,
):
    start_time = time.time()
    sharpe_ratios = []
    policy.eval()
    ref_policy.eval()
    print("graduent_acc_steps", gradient_acc_steps)
    for microbatch_idx in range(gradient_acc_steps):
        global_microbatch = slice_and_move_batch_for_device(
            batch, microbatch_idx, gradient_acc_steps, device=0
        )
        with torch.no_grad():
            policy_chosen_logps, policy_rejected_logps = concatenated_forward(
                policy, global_microbatch
            )
            reference_chosen_logps, reference_rejected_logps = concatenated_forward(
                ref_policy, global_microbatch
            )
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (
            beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        reward_diff = rejected_rewards - chosen_rewards
        pref_p2 = torch.sigmoid(reward_diff).to("cpu")
        pref_p_flip_norm = (1 - 1 / pref_p2).abs()
        if selection_strategy == "sharpe":
            sharp_ratio_mb = (1 + pref_p_flip_norm) / ((1 - pref_p_flip_norm).abs())
        if selection_strategy == "wsharpe":
            pref_p1 = 1 - pref_p2
            means = pref_p1 + pref_p2 * pref_p_flip_norm
            variances = (pref_p1 * ((1 - means) ** 2)) + (
                pref_p2 * (pref_p_flip_norm - means) ** 2
            )
            stds = torch.sqrt(variances)
            sharp_ratio_mb = means / stds
        sharpe_ratios.append(sharp_ratio_mb)
    sharpe_ratios = torch.cat(sharpe_ratios, dim=0)
    values, indices = torch.topk(sharpe_ratios, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    end_time = time.time()
    torch.cuda.empty_cache()
    print(f"Data selection elapsed: {end_time - start_time:.2f}s")
    policy.train()
    return out_batch


def select_apl_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    beta: float = 0.1,
    gradient_acc_steps=12,
):
    start_time = time.time()
    reward_diffs = []
    policy.eval()
    ref_policy.eval()
    for microbatch_idx in range(gradient_acc_steps):
        global_microbatch = slice_and_move_batch_for_device(
            batch, microbatch_idx, gradient_acc_steps, device=0
        )
        with torch.no_grad():
            policy_chosen_logps, policy_rejected_logps = concatenated_forward(
                policy, global_microbatch
            )
            reference_chosen_logps, reference_rejected_logps = concatenated_forward(
                ref_policy, global_microbatch
            )
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (
            beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        reward_diff = (rejected_rewards - chosen_rewards).abs().to("cpu")
        reward_diffs.append(reward_diff)
    reward_diffs = torch.cat(reward_diffs, dim=0)
    values, indices = torch.topk(reward_diffs, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    end_time = time.time()
    torch.cuda.empty_cache()
    print(f"Data selection elapsed: {end_time - start_time:.2f}s")
    policy.train()
    return out_batch


def select_Exp_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    beta: float = 0.1,
    gradient_acc_steps=12,
):
    device = torch.device("cuda")
    correct_grad_norms = []
    swap_grad_norms = []
    policy.eval()
    ref_policy.eval()
    for microbatch_idx in range(gradient_acc_steps):
        global_microbatch = slice_and_move_batch_for_device(
            batch, microbatch_idx, gradient_acc_steps, 0
        )
        local_microbatch = slice_and_move_batch_for_device(global_microbatch, 0, 1, 0)
        with torch.enable_grad():
            policy_chosen_logps, policy_rejected_logps = concatenated_forward(
                policy, local_microbatch
            )
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = concatenated_forward(
                ref_policy, local_microbatch
            )
        correct_losses, correct_chosen_rewards, correct_rejected_rewards = dpo_loss(
            policy_chosen_logps.to(device),
            policy_rejected_logps.to(device),
            reference_chosen_logps.to(device),
            reference_rejected_logps.to(device),
            beta=beta,
            reference_free=False,
        )

        reward_diff = correct_rejected_rewards - correct_chosen_rewards
        pref_p = torch.sigmoid(reward_diff).to("cpu")

        for n in range(len(correct_losses)):
            if n == len(correct_losses) - 1:
                retain_graph = False
            else:
                retain_graph = True
            with torch.enable_grad():
                correct_losses[n].backward(retain_graph=retain_graph)
            total_norm = 0.0
            total_norm_swap = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    p_grad_cpu = p.grad.to("cpu")
                    param_norm = p_grad_cpu.norm(2)
                    total_norm += param_norm.item() ** 2
                    pref_p_flip = 1 - 1 / pref_p[n]
                    swap_grad = p_grad_cpu * pref_p_flip
                    param_norm_swap = swap_grad.norm(2)
                    total_norm_swap += param_norm_swap.item() ** 2
            total_norm = total_norm**0.5
            total_norm_swap = total_norm_swap**0.5
            correct_grad_norms.append(total_norm)
            swap_grad_norms.append(total_norm_swap)
            for param in policy.parameters():
                if param.grad is not None:
                    param.grad.zero_()
    swap_grad_norms = torch.tensor(swap_grad_norms, dtype=torch.float64)
    correct_grad_norms = torch.tensor(correct_grad_norms, dtype=torch.float64)
    means = 0.5 * (correct_grad_norms + swap_grad_norms)
    values, indices = torch.topk(means, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    return out_batch
