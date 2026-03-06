#  COMP0248: Object Detection and Classification Coursework 1
## Hand Gesture Object Detection and Classification

This project is for **COMP0248 Object Detection and Classification**.  
The task is to build a model for **hand gesture understanding** with three outputs at the same time:

- **segmentation mask**
- **gesture classification**
- **bounding box regression**

This repository contains:

- a **baseline RGB model**
- a **lightweight innovation model**
- training, evaluation, and visualisation scripts
- dataset loading code
- saved results and model weights
- a `requirements.txt` file for environment setup

---

## 1. Project structure

```text
project_25050766_Zhen
├── dataset
│   ├── 25050766_Zhen
│   ├── RGB_annotations_only
│   ├── RGB_depth_annotations
│   └── Test data
├── results
│   ├── baseline_rgb
│   ├── innovation_rgb_light
│   ├── figures
│   └── splits
├── src
│   ├── dataloader.py
│   ├── evaluate.py
│   ├── evaluate_innov.py
│   ├── innovation_model.py
│   ├── model.py
│   ├── plot_results.py
│   ├── train.py
│   ├── train_innov.py
│   ├── utils.py
│   ├── visualise.py
│   └── visualise_innov.py
├── weights
│   ├── baseline_rgb_best.pth
│   ├── baseline_rgb_last.pth
│   ├── innovation_rgb_light_best.pth
│   └── innovation_rgb_light_last.pth
├── requirements.txt
└── README.md
```

---

## 2. What is inside this project

### Baseline model
The baseline model is in `src/model.py`.

It is a simple CNN with **three heads**:

- one head for segmentation
- one head for gesture classification
- one head for bounding box prediction

The baseline uses **RGB only**.

### Innovation model
The innovation model is in `src/innovation_model.py`.

This version keeps the model lightweight and does not simply make the whole network much larger.  
The main ideas are:

- **segmentation-guided classification**  
  The predicted segmentation map helps the classifier focus more on the hand area.

- **skip connections in the segmentation decoder**  
  This helps keep more spatial detail for mask prediction.

- **a slightly stronger bounding box head**  
  The bbox branch uses a small MLP instead of only one linear layer.

The innovation model also uses **RGB only**.

---

## 3. Environment

This project was written in Python and PyTorch.

Recommended setup:

- Python 3.9 or 3.10
- PyTorch
- torchvision
- numpy
- pillow
- tqdm

### Install with requirements.txt

A `requirements.txt` file is included in the project root.  
You can install the main dependencies with:

```bash
pip install -r requirements.txt
```

This is the easiest way to prepare the environment.
If you want GPU training, please make sure your PyTorch version matches your CUDA version.

---

## 4. Dataset layout

The code expects the dataset to be placed under the project root like this:

```text
dataset/
├── RGB_depth_annotations/
└── Test data/
    └── COMP0248_Test_data_23/
```

### Training / validation data
The main training data folder is:

```text
dataset/RGB_depth_annotations/
```

Even though the folder name contains `depth`, the current baseline and innovation code both use **RGB only**.

### Test data
The test data folder is:

```text
dataset/Test data/COMP0248_Test_data_23/
```

### Important note
This project uses a **student-level split** for training and validation.  
That means samples are split by person, not by random frame.  
This is important because it helps avoid data leakage.

The split file is saved to:

```text
results/splits/split_seed42.json
```

---

## 5. How the data loader works

The data loader is in `src/dataloader.py`.

For each sample, it returns:

- `image`
- `mask`
- `bbox`
- `label`

The bounding box is computed from the mask.

---

## 6. How to train the baseline model

Go to the `src` folder first:

```bash
cd project_25050766_Zhen/src
```

Then run:

```bash
python train.py --run baseline_rgb
```

This will:

- build the train/val split
- train the baseline model
- save logs to `results/baseline_rgb/`
- save weights to `weights/`

### Useful baseline training arguments

```bash
python train.py --help
```

Common arguments:

- `--epochs`
- `--batch`
- `--lr`
- `--seed`
- `--val_ratio`
- `--run`
- `--num_workers`
- `--w_seg`
- `--w_cls`
- `--w_det`

Example:

```bash
python train.py --run baseline_rgb --epochs 100 --batch 8 --lr 1e-3
```

---

## 7. How to train the innovation model

Go to the `src` folder:

```bash
cd project_25050766_Zhen/src
```

Then run:

```bash
python train_innov.py --run innovation_rgb_light
```

This will:

- load the same dataset
- use the innovation model in `innovation_model.py`
- train with RGB only
- save logs to `results/innovation_rgb_light/`
- save weights to `weights/`

### Useful innovation training arguments

```bash
python train_innov.py --help
```

Common arguments:

- `--epochs`
- `--batch`
- `--lr`
- `--seed`
- `--val_ratio`
- `--run`
- `--num_workers`
- `--base`
- `--guide_alpha`
- `--w_seg`
- `--w_cls`
- `--w_det`
- `--seg_pos_weight`
- `--dice_weight`
- `--patience`
- `--min_delta`

Example:

```bash
python train_innov.py --run innovation_rgb_light --epochs 100 --batch 8 --lr 1e-3
```

---

## 8. How to evaluate the baseline model

To evaluate on the validation split:

```bash
python evaluate.py --run baseline_rgb --split val
```

To evaluate on the test split:

```bash
python evaluate.py --run baseline_rgb --split test
```

The results will be saved in:

```text
results/baseline_rgb/
```

---

## 9. How to evaluate the innovation model

To evaluate on the validation split:

```bash
python evaluate_innov.py --run innovation_rgb_light --split val
```

To evaluate on the test split:

```bash
python evaluate_innov.py --run innovation_rgb_light --split test
```

The results will be saved in:

```text
results/innovation_rgb_light/
```

---

## 10. How to visualise the predictions

### Baseline visualisation

Validation split:

```bash
python visualise.py --run baseline_rgb --split val --num 40
```

Test split:

```bash
python visualise.py --run baseline_rgb --split test --num 40
```

### Innovation visualisation

Validation split:

```bash
python visualise_innov.py --run innovation_rgb_light --split val --num 40
```

Test split:

```bash
python visualise_innov.py --run innovation_rgb_light --split test --num 40
```

The overlay images will be saved under:

```text
results/<run_name>/overlays/
```

These images show:

- ground-truth mask
- predicted mask
- ground-truth box
- predicted box

---

## 11. Output files

### In `weights/`
This folder stores model checkpoints.

Examples:

- `baseline_rgb_best.pth`
- `innovation_rgb_light_best.pth`

### In `results/<run_name>/`
This folder stores:

- training log
- evaluation metrics
- visualisation overlays

### In `results/splits/`
This folder stores the saved train/validation split file.

---

## 12. Metrics used in this project

This project reports these metrics:

- **mean segmentation IoU**
- **mean Dice score**
- **bounding box IoU**
- **detection accuracy at IoU 0.5**
- **top-1 classification accuracy**
- **macro F1 score**

---

## 13. Reproducibility notes

To reproduce the same result as closely as possible:

1. use the same dataset folder structure  
2. install dependencies from `requirements.txt`  
3. keep the same random seed  
4. use the same train/val split  
5. use the same run name and checkpoint selection  
6. make sure the package versions are close

Recommended seed:

```bash
--seed 42
```

Please note that training speed may be different on different GPUs, CUDA versions, or PyTorch versions.

---

## 14. Typical workflow

A simple full workflow looks like this:

### Step 1: install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: train the baseline

```bash
cd src
python train.py --run baseline_rgb
```

### Step 3: evaluate the baseline

```bash
python evaluate.py --run baseline_rgb --split val
python evaluate.py --run baseline_rgb --split test
```

### Step 4: visualise the baseline

```bash
python visualise.py --run baseline_rgb --split val --num 40
```

### Step 5: train the innovation model

```bash
python train_innov.py --run innovation_rgb_light
```

### Step 6: evaluate the innovation model

```bash
python evaluate_innov.py --run innovation_rgb_light --split val
python evaluate_innov.py --run innovation_rgb_light --split test
```

### Step 7: visualise the innovation model

```bash
python visualise_innov.py --run innovation_rgb_light --split val --num 40
```
---