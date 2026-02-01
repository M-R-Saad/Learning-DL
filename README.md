# Deep Learning From Scratch

A collection of deep learning algorithms implemented from scratch using only NumPy. This repository was developed as part of my Deep Learning course, where I learned the mathematical foundations of each algorithm and then implemented them to solidify my understanding.

## Overview

The goal of this repository is to understand how deep learning algorithms work at a fundamental level — no PyTorch, no TensorFlow, just pure NumPy and math. Each notebook includes:

- Implementation of forward and backward passes
- Training loops with gradient descent
- Visualizations of results
- Example datasets for testing

## Algorithms Implemented

| # | Notebook | Algorithm | Key Concepts |
|---|----------|-----------|--------------|
| 01 | `01_MLP_from_scratch.ipynb` | Multi-Layer Perceptron | Feedforward networks, backpropagation, activation functions |
| 02 | `02_SOM_from_scratch.ipynb` | Self-Organizing Maps | Competitive learning, neighborhood functions, unsupervised clustering |
| 03 | `03_SemiSupervised_from_scratch.ipynb` | Semi-Supervised Learning | Label propagation, pseudo-labeling, leveraging unlabeled data |
| 04 | `04_GRNN_from_scratch.ipynb` | General Regression Neural Network | Radial basis functions, kernel-based regression |
| 05 | `05_RNN_from_scratch.ipynb` | Recurrent Neural Network | Sequential data, hidden states, BPTT |
| 06 | `06_LSTM_from_scratch.ipynb` | Long Short-Term Memory | Forget/input/output gates, cell state, vanishing gradient solution |
| 07 | `07_CNN_from_scratch.ipynb` | Convolutional Neural Network | Convolution, pooling, feature maps |
| 08 | `08_GNN_from_scratch.ipynb` | Graph Neural Network | Message passing, node embeddings, adjacency matrices |
| 09 | `09_DBN_from_scratch.ipynb` | Deep Belief Network | Restricted Boltzmann Machines, layer-wise pretraining |
| 10 | `10_GAN_from_scratch.ipynb` | Generative Adversarial Network | Generator, discriminator, adversarial training |
| 11 | `11_Transformer_from_scratch.ipynb` | Transformer | Self-attention, positional encoding, multi-head attention |

## Requirements

```
numpy
matplotlib
```

## Usage

1. Clone or download this repository
2. Open any notebook in Jupyter or VS Code
3. Run the cells sequentially to see the implementation and results

## Project Structure

```
Learning DL/
├── 01_MLP_from_scratch.ipynb
├── 02_SOM_from_scratch.ipynb
├── 03_SemiSupervised_from_scratch.ipynb
├── 04_GRNN_from_scratch.ipynb
├── 05_RNN_from_scratch.ipynb
├── 06_LSTM_from_scratch.ipynb
├── 07_CNN_from_scratch.ipynb
├── 08_GNN_from_scratch.ipynb
├── 09_DBN_from_scratch.ipynb
├── 10_GAN_from_scratch.ipynb
├── 11_Transformer_from_scratch.ipynb
└── README.md
```

## What I Learned

- How gradient descent and backpropagation actually work mathematically
- The intuition behind different architectures (CNNs for spatial data, RNNs for sequences, etc.)
- Why certain techniques like batch normalization and dropout help training
- The differences between various activation functions and when to use them
- How attention mechanisms revolutionized sequence modeling

## Acknowledgments

Based on lectures and mathematical examples from my Deep Learning course.
