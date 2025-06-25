# 🤖 Distributed Transformers Training on Azure ML with Accelerate

This repository contains a complete **training pipeline** for transformer models using **Hugging Face Accelerate**, **PyTorch**, and **Azure Machine Learning Pipelines**.  
The training and evaluation processes are performed on **datasets stored in Azure Blob Storage**, using a **custom Docker image** and (optionally) a **GPU-enabled compute cluster**.

---

## 🚀 Features

- Training with Hugging Face `accelerate` and `transformers.Trainer`  
- Pre-training and post-training evaluation using `scikit-learn`'s `classification_report`  
- Custom Docker image with CUDA support (via `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`)  
- Datasets loaded from Azure Blob Storage using `URI_FOLDER`  
- Containerized logic with `train.py`, `Dockerfile`, and `config_accelerate.yaml`  
- Compatible with both CPU and GPU clusters on Azure ML  
- Outputs saved to the `output_dir`, including the final model and tokenizer  

---

## ☁️ Launch the pipeline

```bash
python run_pipeline.py
```

---
