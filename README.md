# Medical Q&A Fine-Tuning with Gemma 2B

This repository contains a Colab notebook that fine-tunes the [`google/gemma-2b`](https://huggingface.co/google/gemma-2b) model on a custom Medical Q&A dataset using QLoRA.

## Features
- Instruction-tuned format: User ↔ Model style
- Uses 4-bit quantization via `BitsAndBytes` for memory efficiency
- LoRA (QLoRA) for parameter-efficient fine-tuning
- Full training + inference + export pipeline

## Folder Structure
- `gemma_2b_medical_qa_finetuning.ipynb` - Colab notebook with training pipeline
- `requirements.txt` - Python packages needed
- `README.md` - Project documentation

## Inference Example
```python
generate_answer("What are the symptoms of diabetes?")
