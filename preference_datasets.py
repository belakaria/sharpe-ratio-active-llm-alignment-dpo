from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets
import torch
import tqdm
from bs4 import BeautifulSoup, NavigableString
from torch.nn.utils.rnn import pad_sequence


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, "html.parser")

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == "p":
            text.append(
                "".join(
                    child.string
                    for child in element.children
                    if isinstance(child, NavigableString)
                )
            )
        elif element.name == "pre":
            for code in element.find_all("code"):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == "code":
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_shp(
    split: str, silent: bool = False, cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

    We filter preference pairs to only keep pairs where the score ratio is at least 2.
    For this dataset, the sft_target is the response with the highest score.
    """
    print(f"Loading SHP dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset("stanfordnlp/SHP", split=split, cache_dir=cache_dir)

    data = defaultdict(lambda: defaultdict(list))

    for row in tqdm.tqdm(dataset, desc="Processing SHP", disable=silent):
        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        responses = [" " + row["human_ref_A"], " " + row["human_ref_B"]]
        scores = [row["score_A"], row["score_B"]]
        if prompt in data:
            n_responses = len(data[prompt]["responses"])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]["pairs"].append(
            (n_responses, n_responses + 1)
            if row["labels"] == 1
            else (n_responses + 1, n_responses)
        )
        data[prompt]["responses"].extend(responses)
        data[prompt]["scores"].extend(scores)

    for prompt in data:
        data[prompt]["sft_target"] = max(
            data[prompt]["responses"],
            key=lambda x: data[prompt]["scores"][data[prompt]["responses"].index(x)],
        )
        del data[prompt]["scores"]

    return data


def get_hh(
    split: str, silent: bool = False, cache_dir: str = None
) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt1': {
            'responses': List[str],
            'pairs': List[Tuple[int, int]],
            'sft_target': str
        },
        'prompt2': {
            ...
        },
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.

    For this dataset, the sft_target is just the chosen response.
    """
    print(f"Loading HH dataset ({split} split) from Huggingface...")
    dataset = datasets.load_dataset(
        "Anthropic/hh-rlhf", split=split, cache_dir=cache_dir
    )
    # print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex["chosen"])
        chosen_response = ex["chosen"][len(prompt) :]
        rejected_response = ex["rejected"][len(prompt) :]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing HH", disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend(responses)
        data[prompt]["sft_target"] = chosen

    return data


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == "shp":
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == "hh":
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {
        "responses",
        "pairs",
        "sft_target",
    }, f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data


def get_collate_fn(
    tokenizer,
) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

    The collate function takes a list of examples (dicts, where values are lists of
      ints [tokens] or strings [the original texts]) and returns a batch of examples,
      PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if (
                    "prompt" in k
                ):  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if (
                    "prompt" in k
                ):  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn


def tokenize_batch_element(
    prompt: str,
    chosen: str,
    rejected: str,
    truncation_mode: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
) -> Optional[Dict]:
    """Tokenize a single batch element.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
      in case the prompt + chosen or prompt + rejected responses is/are too long. First
      we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
      the sum of the length of the prompt and the chosen/rejected response, with -100 for the
      prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    if tokenizer.eos_token_id in prompt_tokens["input_ids"]:
        print(f"Prompt contains EOS token: {prompt}")
        return None
    if tokenizer.eos_token_id in chosen_tokens["input_ids"]:
        print(f"Chosen response contains EOS token: {chosen}")
        return None
    if tokenizer.eos_token_id in rejected_tokens["input_ids"]:
        print(f"Rejected response contains EOS token: {rejected}")
        return None

    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(
        len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
    )
    # print("prompt,response,rejected",len(prompt_tokens['input_ids']),len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == "keep_end":
            prompt_tokens = {
                k: v[-max_prompt_length:] for k, v in prompt_tokens.items()
            }
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        chosen_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()
        }
        rejected_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()
        }

    # Create labels
    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
    }
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
        -100
    ] * len(prompt_tokens["input_ids"])

    batch = {}

    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    batch["rejected"] = prompt + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    return batch


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != " " and str_b[idx] != " ":
                return False
            else:
                if str_a[idx] == " ":
                    str_a = str_a[:idx] + str_a[idx + 1 :]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1 :]

    return True
