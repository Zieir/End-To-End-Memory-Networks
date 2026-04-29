# End-To-End Memory Networks (MemN2N)

This repository contains a PyTorch implementation of the **End-To-End Memory Network** (MemN2N) architecture, as proposed by Sukhbaatar et al. (2015). The model is designed to perform multi-hop logical reasoning over an external memory buffer, specifically evaluated on the Facebook bAbI question-answering dataset.

## Authors

- **KERMADJ Zineddine** — Université Paris-Saclay  
- **BEURTHERET Eloi** — Université Paris-Saclay

---

## Project Overview

This implementation focuses on a differentiable attention model that can be trained with weak supervision (only the final answer is required, not the supporting facts). By executing multiple **memory hops** over the input memory, the network learns to perform complex chains of reasoning.

### Key Technical Features

- **Multi-Hop Attention**  
  Recurrent memory processing with **K = 3** computational hops.

- **Weight Tying**  
  Implementation of the **Adjacent** weight-tying scheme:

  ```math
  A^{k+1} = C^k
  ```

  This improves parameter efficiency and aids generalization.

- **Bag-of-Words (BoW)**  
  Baseline sentence representation.

- **Position Encoding (PE)**  
  Advanced embedding method to preserve syntactic word order within sentences.

- **Linear Start (LS)**  
  Training strategy to prevent poor local minima by removing softmax during early epochs.

- **Temporal Encoding**  
  Learned embeddings that track the relative chronological order of facts.

---

## Repository Structure

```text
.
├── model.py
├── train.py
├── eval.py
├── run_all.py
└── data/
    └── dataloader.py
```

### File Descriptions

- **`model.py`**  
  Core MemN2N architecture and data tensor preparation.

- **`train.py`**  
  Training script with:
  - Learning-rate annealing
  - Gradient clipping
  - Task-specific optimization

- **`eval.py`**  
  Modular inference logic for testing saved model checkpoints.

- **`run_all.py`**  
  Automated pipeline to train and benchmark all 20 bAbI tasks sequentially.

- **`data/dataloader.py`**  
  Custom data loader for automated downloading and preprocessing of the bAbI dataset.

---

## Usage

### 1. Train a Single Task

```bash
python train.py [task_id]
```

Example (Task 1):

```bash
python train.py 1
```

Example (Task 2):

```bash
python train.py 2
```

---

### 2. Evaluate a Saved Model

```bash
python eval.py [task_id]
```

Example:

```bash
python eval.py 1
```

---

### 3. Run the Full Benchmark (All 20 Tasks)

```bash
python run_all.py
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

---

## Reference

**Paper:**  
Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus.  
*End-To-End Memory Networks* (2015)

```bibtex
@article{sukhbaatar2015memn2n,
  title={End-To-End Memory Networks},
  author={Sukhbaatar, Sainbayar and Szlam, Arthur and Weston, Jason and Fergus, Rob},
  journal={Advances in Neural Information Processing Systems},
  year={2015}
}
```

---

## Notes

- **BoW** serves as a baseline and struggles on tasks requiring positional reasoning.
- **Position Encoding (PE)** addresses these structural limitations.
- **Linear Start** improves optimization stability.
- **Adjacent Weight Tying** reduces parameter count while preserving multi-hop reasoning power.

---

## License

This project is provided for research and educational purposes.
