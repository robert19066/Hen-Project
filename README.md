# Hen AI

Hen is a fine-tunable, Qwen-powered language model designed for interactive chat and text generation. Hen comes in multiple variants including Hen-2B, Hen-o1, and Hen-o2, with Hen-o2 featuring advanced reasoning, Romanian language support, and anti-overfitting mechanisms.

# Features

- tilingual chat support (English & Romanian)

- Fast training and fine-tuning with LoRA

- Modular scripts for training and chatting

- Handles text-based queries,and creative prompts

- Lightweight 4-bit model loading for efficiency

- Supports both single-line and multi-line chat modes

# Model Variants
Hen-o1 (Hen-4B)	Full 4-billion parameter model; trained on Alpaca, OpenAssistant, and Dolly
Hen-o2 (Hen-4B)	Advanced 4-billion parameter model; anti-overfitting, reasoning improvements, and Romanian support
Hen-o2 MAX	Model variant with memory and optimized performance
# Scripts
## `hen_trainer.py`

Use this script to train or fine-tune Hen on custom datasets. Features include:

- 4-bit model loading for efficiency

- Dataset filtering, formatting, and merging

- LoRA fine-tuning

- Real-time training loss monitoring

`python hen_trainer.py --dataset path/to/your/dataset`

## `hen_chat.py`

Use this script for interactive chatting with Hen. Features include:

- Single-line and multi-line chat modes

- Romanian translation support (toggleable)

- Keeps conversation history for context

- Configurable temperature and model variant

`python hen_chat.py`


### Keyboard Shortcuts:

- Ctrl+O — Swap between single-line and multi-line input

- Ctrl+R — Toggle Romanian mode

- Ctrl+C — Exit or open menu

# Requirements

- Python 3.10+

- PyTorch

- Transformers

- Datasets

- PEFT (Parameter-Efficient Fine-Tuning)

- googletrans library for Romanian translation

# Install dependencies:

`pip install torch transformers peft googletrans==4.0.0rc1`

# Notes

- Hen-o2 is highly recommended for reasoning and creative tasks.

- Hen-o2 MAX will include memory for context persistence.

- 4-bit model loading reduces GPU memory requirements, making Hen accessible on consumer GPUs.

# License

This project is licensed under the Apache License 2.0
.