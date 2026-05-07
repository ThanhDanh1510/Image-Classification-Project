# TrashNet Image Classification

## Overview

This repository implements a TrashNet image-classification workflow using:

- `ResNet50` pretrained on ImageNet
- separate train and evaluation transforms
- ImageNet normalization
- train/validation/test split of `60/20/20`
- mixed precision training when CUDA is available
- early stopping on validation loss
- DVC-managed data, training, and evaluation stages
- Streamlit-based inference demo

## Repository Structure

```text
.
|-- app/
|   `-- streamlit_app.py
|-- artifacts/
|   |-- model/
|   `-- reports/
|-- data/
|   |-- external/
|   |-- raw/
|   `-- processed/
|-- notebook/
|   `-- final_notebook.ipynb
|-- scripts/
|   |-- prepare_data.py
|   |-- train.py
|   `-- evaluate.py
|-- src/image_classification/
|   |-- config.py
|   |-- data.py
|   |-- evaluate.py
|   |-- inference.py
|   |-- model.py
|   `-- training.py
|-- dvc.yaml
|-- params.yaml
`-- pyproject.toml
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\dvc.exe repro
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

## Environment Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it on Windows PowerShell

If PowerShell blocks local scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

If you prefer not to activate the environment, use the executables directly from `.venv\Scripts\`.

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

## GPU Support

The training code automatically uses CUDA when available and when the installed PyTorch build includes CUDA support.

### Verify your PyTorch installation

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If you see a version ending in `+cpu`, or `torch.cuda.is_available()` returns `False`, your environment is running CPU-only PyTorch.

### Install a CUDA-enabled PyTorch build

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Verify GPU detection

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

During training, the project prints runtime diagnostics including:

- PyTorch version
- configured device
- resolved device
- CUDA availability
- CUDA build
- GPU count
- AMP status

If training falls back to CPU, the script prints a warning before the first epoch.

## Configuration

Project configuration is defined in [params.yaml](C:/Users/PC/Downloads/Image-Classification/params.yaml).

Current defaults include:

- `train_split: 0.6`
- `val_split: 0.2`
- `test_split: 0.2`
- `batch_size: 64`
- `epochs: 50`
- `learning_rate: 0.00005`
- `patience: 3`
- `device: "auto"`
- `amp: true`
- `multi_gpu: true`

To force GPU usage explicitly:

```yaml
train:
  device: "cuda"
```

## DVC Workflow

### Initialize DVC

If the repository has not been initialized with DVC yet:

```powershell
.\.venv\Scripts\dvc.exe init
```

If `.dvc/` already exists, skip this step.

### Run individual stages

Prepare the dataset:

```powershell
.\.venv\Scripts\dvc.exe repro prepare
```

Train the model:

```powershell
.\.venv\Scripts\dvc.exe repro train
```

Evaluate the model:

```powershell
.\.venv\Scripts\dvc.exe repro evaluate
```

### Run the full pipeline

```powershell
.\.venv\Scripts\dvc.exe repro
```

### Useful DVC commands

View the pipeline graph:

```powershell
.\.venv\Scripts\dvc.exe dag
```

Inspect metrics:

```powershell
.\.venv\Scripts\dvc.exe metrics show
```

## Running Without DVC

For direct script execution:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_data --params params.yaml
.\.venv\Scripts\python.exe -m scripts.train --params params.yaml
.\.venv\Scripts\python.exe -m scripts.evaluate --params params.yaml
```

## Streamlit Demo

After training has produced a checkpoint:

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

The demo loads:

- `artifacts/model/best_model.pt`
- `artifacts/model/class_names.json`

It then:

- preprocesses the uploaded image
- runs inference with the trained model
- shows the top predicted class
- shows ranked model scores for the top classes

## Generated Artifacts

After training and evaluation, the main outputs are:

- `artifacts/model/best_model.pt`
- `artifacts/model/class_names.json`
- `artifacts/model/training_history.json`
- `artifacts/model/training_summary.json`
- `artifacts/reports/metrics.json`
- `artifacts/reports/accuracy_per_class.png`
- `artifacts/reports/confusion_matrix.png`

## Platform Notes

### Windows

- The codebase forces `num_workers=0` on Windows to avoid common local `DataLoader` hangs.
- If PowerShell blocks activation scripts, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- If DVC reports stale lock warnings after an interrupted run, rerunning `dvc repro train` is usually sufficient.

### Colab and Kaggle

The notebook version is environment-aware and supports both Colab and Kaggle paths. The Python codebase is intended primarily for local and scripted execution, with DVC handling the reproducible workflow.

## Mapping From Notebook to Codebase

- dataset bootstrap, transforms, splits, and loaders: [src/image_classification/data.py](C:/Users/PC/Downloads/Image-Classification/src/image_classification/data.py)
- model definition and training hooks: [src/image_classification/model.py](C:/Users/PC/Downloads/Image-Classification/src/image_classification/model.py)
- early stopping, AMP, and training loop: [src/image_classification/training.py](C:/Users/PC/Downloads/Image-Classification/src/image_classification/training.py)
- evaluation reports and metrics: [src/image_classification/evaluate.py](C:/Users/PC/Downloads/Image-Classification/src/image_classification/evaluate.py)
- inference and demo app: [src/image_classification/inference.py](C:/Users/PC/Downloads/Image-Classification/src/image_classification/inference.py), [app/streamlit_app.py](C:/Users/PC/Downloads/Image-Classification/app/streamlit_app.py)

