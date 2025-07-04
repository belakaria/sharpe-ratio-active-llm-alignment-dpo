# Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF

## Overview

This repository contains the official implementation of **SHARP-DPO** and **W-SHARP-DPO**, methods we proposed in the [Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF](https://arxiv.org/abs/2503.22137) paper.

These methods extend the Direct Preference Optimization (DPO) pipeline by introducing **active selection** strategies that reduce the number of preference queries needed during training based on the sharpe ratio.

---

## What's in this repo?

The DPO pipeline we build upon has two main stages:

1. **Supervised fine-tuning (SFT)** on an initial dataset.
2. **Active preference optimization** to refine the model using preference pairs.

Our codebase is structured similarly to the [original DPO implementation](https://github.com/eric-mitchell/direct-preference-optimization).

---

### Key Components

- `train.py`: Main entry point for training across all methods. Supports both base and QLoRA-based models.
- `trainers.py`: Training loop logic, including batch-wise active data selection.
- `data_selection.py`: Implementations of SHARP, W-SHARP, and [APL](https://arxiv.org/pdf/2402.08114) strategies. We also include EXP (Expectation of gradient), this is only for the purpose of ablation study, we do not recommend using it, it is very slow and memory consuming.
- `preference_datasets.py`: Dataset loading. **To train on your own data, modify this file.**
- `utils.py`: Shared utilities.
- `config/`: Configuration files. Variables may be set in config files or via command-line flags.

---

## Running Active Learning algorithms (SHARP-DPO, W-SHARP-DPO, APL)

These methods simulate active learning on **pre-collected preference data**. To ensure a realistic evaluation, data used for SFT is excluded from the active learning phase using the `pretrain_fraction` parameter.

```bash
python -u train.py model=gpt2-large datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=sft_policy exp_name=hh_sharp_gpt2-large \
  gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 \
  trainer=BasicTrainer sample_during_eval=true pretrain_fraction=0.3  \
  active=true qlora=true selection_strategy=sharpe selection_ratio=6
```

```bash
python -u train.py model=gpt2-large datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=sft_policy exp_name=hh_wsharp_gpt2-large \
  gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 \
  trainer=BasicTrainer sample_during_eval=true pretrain_fraction=0.3 \
  active=true qlora=true selection_strategy=wsharpe selection_ratio=6
```

---

> **Note:** Some environment variables must be set depending on the method and configuration you are running:
>
> * `HF_TOKEN`: Required for downloading certain Hugging Face models or datasets.
> * `WANDB_API_KEY`: Required if Weights & Biases (`wandb`) is enabled in the config file.

---

## Citing

If you find this work helpful in your research or applications, please consider citing:

```bibtex
@article{belakaria2025sharpe,
  title={Sharpe Ratio-Guided Active Learning for Preference Optimization in RLHF},
  author={Belakaria, Syrine and Kazdan, Joshua and Marx, Charles and Cundy, Chris and Neiswanger, Willie and Koyejo, Sanmi and Engelhardt, Barbara E and Ermon, Stefano},
  journal={arXiv preprint arXiv:2503.22137},
  year={2025},
  url     = {https://arxiv.org/abs/2503.22137}
}
```

---

## Acknowledgments

This project builds upon the [Direct Preference Optimization (DPO)](https://github.com/eric-mitchell/direct-preference-optimization) framework by Eric Mitchell et al. We thank the authors for their open-source contributions.