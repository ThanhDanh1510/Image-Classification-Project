# Image Classification Project

Du an nay duoc tach tu `notebook/final_notebook.ipynb` thanh cau truc Python chuan, co DVC de quan ly pipeline du lieu/huan luyen/danh gia va Streamlit de demo model.

## Cau truc

```text
.
|-- data/
|   |-- external/
|   |-- raw/
|   `-- processed/
|-- artifacts/
|   |-- model/
|   `-- reports/
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
|-- app/
|   `-- streamlit_app.py
|-- dvc.yaml
|-- params.yaml
`-- pyproject.toml
```

## Cai dat

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

Neu ban chua khoi tao DVC trong repo, chay them:

```powershell
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## Chay pipeline bang DVC

1. Cap nhat tham so trong `params.yaml` neu can.
2. Tai va chuan hoa du lieu:

```powershell
dvc repro prepare
```

3. Huan luyen model:

```powershell
dvc repro train
```

4. Danh gia model:

```powershell
dvc repro evaluate
```

Hoac chay toan bo:

```powershell
dvc repro
```

Khi thay doi `params.yaml` hoac code trong `src/`, DVC se tu dong chi chay lai stage can thiet.

## Theo doi artifact va cache

- `data/processed` duoc sinh ra boi stage `prepare`
- `artifacts/model/best_model.pt` la `state_dict` cua ResNet50 theo logic notebook
- `artifacts/model/training_history.json` luu lich su train voi cac key `Train Loss`, `Validation Loss`, `Validation Accuracy`
- `artifacts/reports/metrics.json` chua overall accuracy va metric theo tung class
- `artifacts/reports/accuracy_per_class.png` va `confusion_matrix.png` la cac report sinh tu stage evaluate

Ban co the xem DAG:

```powershell
dvc dag
```

Va so sanh thay doi metric:

```powershell
dvc metrics show
git checkout <commit-cu>
dvc metrics diff <commit-moi>
```

## Demo voi Streamlit

Sau khi da train xong:

```powershell
streamlit run app/streamlit_app.py
```

App se load:

- `artifacts/model/best_model.pt`
- `artifacts/model/class_names.json`

Ban upload anh, app se resize ve dung kich thuoc va hien thi top prediction cung cac score cua model.

## Mapping tu final_notebook

- Chuan bi TrashNet + tách transform train/eval + split `60/20/20` -> `src/image_classification/data.py`
- ResNet50 pretrained + sigmoid output + DataParallel -> `src/image_classification/model.py`
- AMP + early stopping + train history -> `src/image_classification/training.py`
- Metric theo class + confusion matrix % + accuracy per class -> `src/image_classification/evaluate.py`
- Suy luan va Streamlit demo -> `src/image_classification/inference.py`, `app/streamlit_app.py`

## Ghi chu

- Logic huan luyen trong codebase duoc dong bo theo `final_notebook.ipynb`.
- Tren Windows, DataLoader trong codebase tu dong dung `num_workers=0` de tranh loi/treo khi chay local.
