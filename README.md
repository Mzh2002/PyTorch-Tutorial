# PyTorch Tutorial for Beginners

A structured, folder-based course that takes you from absolute basics to a full end-to-end deep learning project — all in plain Python scripts (no notebooks, no AI APIs).

## Course Structure

| Lesson | Topic |
|--------|-------|
| 01 | Tensors, shapes, dtype, device |
| 02 | Tensor operations and broadcasting |
| 03 | Autograd and gradients |
| 04 | Linear regression from scratch |
| 05 | nn.Module, parameters, and layers |
| 06 | Loss functions and optimizers |
| 07 | Building a training loop |
| 08 | Dataset and DataLoader |
| 09 | Classification with MLP |
| 10 | CNN basics |
| 11 | Image classification project |
| 12 | Saving/loading models and inference script |
| 13 | Evaluation metrics and confusion matrix |
| 14 | Full project: train, evaluate, save, and run inference |

Each lesson folder contains three files:

- **`tutorial.py`** — teaches the concept with runnable examples and clear comments.
- **`homework.py`** — gives exercises for the learner to complete (look for `TODO` markers).
- **`answer.py`** — provides the full solution to the homework.

## Prerequisites

- Python 3.9+
- pip

## Setup

```bash
# Clone the repository
git clone https://github.com/Mzh2002/PyTorch-Tutorial.git
cd PyTorch-Tutorial

# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running a Lesson

```bash
# Run a tutorial
python lesson_01_tensors/tutorial.py

# Try the homework (fill in the TODOs first!)
python lesson_01_tensors/homework.py

# Check against the answer
python lesson_01_tensors/answer.py
```

## Tips for Learners

1. **Run each `tutorial.py` first** — read the comments and observe the printed output.
2. **Try `homework.py` on your own** — fill in the `TODO` sections before looking at the answer.
3. **Compare with `answer.py`** — understand why the solution works.
4. **Experiment!** — change values, add print statements, break things on purpose to learn.

## Datasets

- Lessons 01–03 use small tensors for demonstrating PyTorch mechanics (no download required).
- Lessons 04–08 use **real datasets from scikit-learn**:
  - **California Housing** — housing prices from the 1990 US Census (fetched from the internet on first run).
  - **Diabetes** — disease progression from 442 diabetes patients.
  - **Wine** — chemical analysis of 178 wines from three cultivars.
  - **Iris** — measurements of 150 iris flowers from three species.
- Lessons 09–14 use **FashionMNIST** which is automatically downloaded by `torchvision` on first run (~30 MB).

## Project Layout

```
PyTorch-Tutorial/
├── README.md
├── requirements.txt
├── lesson_01_tensors/
│   ├── tutorial.py
│   ├── homework.py
│   └── answer.py
├── lesson_02_operations/
│   ├── ...
├── ...
└── lesson_14_full_project/
    ├── tutorial.py
    ├── homework.py
    └── answer.py
```

## License

This project is provided for educational purposes. Feel free to use, modify, and share.
