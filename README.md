# TraGT

TraGT is a project that integrates graph-based and sequence-based models for various datasets. This repository contains implementations of transformer-based models, graph neural networks, and fusion models for tasks like classification and reconstruction.

## Overview

The TraGT project aims to combine graph-based and sequence-based data representations to improve performance on various machine learning tasks. The project leverages:

- **Graph Neural Networks (GNNs)** for processing graph-structured data.
- **Transformer-based models** for handling sequential data.
- **Fusion models** to integrate graph and sequence embeddings for downstream tasks.

The project supports multiple datasets and provides flexibility in enabling or disabling specific components of the model pipeline.

---

## Requirements

To run the project, ensure you have the following installed:

- Python 3.12.3 or later
- PyTorch
- Torch Geometric
- NumPy
- Scikit-learn
- Matplotlib

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Prepare the Dataset

Place your dataset files in the `input/` directory. Ensure the dataset is formatted correctly as required by the model.

### 2. Run the Program

To reproduce the results, navigate to the desired version directory (e.g., `TraGT_v4_new_arch`) and run the `main.py` script in the terminal:


Then execute the script:

```bash
python src/TraGT_v3/main.py
```

### 3. Provide Arguments

The `main.py` script accepts arguments to specify the dataset and model configuration. For example:

```bash
python main.py --data_name bace --options [True, True, True, True]
```

- `--data_name`: Specifies the dataset to use (e.g., `bace`, `bbbp`, `fda`, `logp`).
- `--options`: A list of boolean values to enable or disable specific components:
  - `options[0]`: Enable graph model.
  - `options[1]`: Enable sequence model.
  - `options[2]`: Enable fusion model.
  - `options[3]`: Enable reconstruction loss (if applicable).

### 4. View Results

- Training and testing accuracy details are saved in the `output/` directory.
- Trained models are saved in the `saved_models/` directory.

---

## Example Command

To train the model on the `bace` dataset with all components enabled:

```bash
python main.py --data_name bace --options [True, True, True, True]
```

---

## Visualizations

- Use the `loss_plot.ipynb` and `tsnp_plot.ipynb` notebooks to visualize training loss and t-SNE plots.


For any questions or issues, please contact the project maintainers.

