# Graph-to-Sequence for the Traveling Salesman Problem

This repository contains an implementation of the Graph-to-Sequence model for solving the Traveling Salesman Problem (TSP). The Graph-to-Sequence model is a neural network architecture that takes a graph representation of the TSP and outputs a sequence of nodes representing a tour of the graph.

## Requirements

To run this code, you will need the following dependencies:

- Python 3.6 or later
- PyTorch 1.7 or later
- NumPy
- Matplotlib

## Usage

### Training

To train the Graph-to-Sequence model on TSP instances, use the following command:

```bash
python train.py --graph-size 20 --batch-size 128 --epochs 100 --lr 0.001
```

This command will train the model on TSP instances with 20 nodes. You can adjust the parameters like batch size, epochs, and learning rate as needed.

### Evaluation

To evaluate the trained model on a TSP instance, use the following command:

```bash
python evaluate.py --graph-size 20 --model-path models/model.pt --data path_to_dataset
```

Here, provide the appropriate model path and dataset path. The model will be evaluated on TSP instances with 20 nodes.

## Acknowledgements

This implementation is based on the following paper:

- **Attention, Learn to Solve Routing Problems!**
  - **Authors:** Wouter Kool, Herke van Hoof, Max Welling
  - **Year:** 2019
  - **Month:** February
  - **URL:** [http://arxiv.org/abs/1803.08475](http://arxiv.org/abs/1803.08475)
  - **DOI:** [10.48550/arXiv.1803.08475](https://doi.org/10.48550/arXiv.1803.08475)