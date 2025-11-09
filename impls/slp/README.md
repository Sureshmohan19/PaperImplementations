# Single Layer Perceptron (SLP)

A from-scratch implementation of the single layer perceptron, one of the foundational concepts in neural networks.

## Quick Start

**Prerequisites:** 
- Mamba installed (see main README)
- Weights & Biases account (free at [wandb.ai](https://wandb.ai))

```bash
# 1. Create environment
mamba env create -f environment.yml

# 2. Login to W&B (first time only)
mamba run -n slp wandb login

# 3. Run training
mamba run -n slp python train.py
```

**First time setup:**
When you run `wandb login`, it will open a browser and give you an API key. Paste it in the terminal.

**What you'll see:**
- Training progress in terminal
- Real-time metrics & visualizations at wandb.ai dashboard
- Interactive plots (zoom, pan, compare runs)
- System metrics (GPU/CPU usage, memory)

## What is a Single Layer Perceptron?

A perceptron is a computational model of a neuron that tries to decide between two classes (binary classification). It receives several inputs — each multiplied by a weight that determines its importance — adds them all together (plus a bias term), and applies a **step function** to produce a binary output (0 or 1).

**This implementation uses ONLY the step function** (the original perceptron from Rosenblatt, 1958). Sigmoid and other continuous activations will be introduced in the MLP implementation where they conceptually belong (with gradient descent and backpropagation).

### Visual Representation

```
inputs (x1, x2, ..., xn)
   │
   │  weighted by w1, w2, ..., wn
   │  bias term b
   │
   ▼
  [Σ w_i * x_i + b]  → activation function  →  binary output (y)
```

### Mathematical Representation

```
y = f(Σ w_i * x_i + b)
```

Where:
- `f` is the activation function
- `w_i` are the weights
- `x_i` are the inputs
- `b` is the bias term

## Activation Function: Step Function Only

This implementation uses the **step function** exclusively - the activation from the original perceptron algorithm (Rosenblatt, 1958).

```
f(y) = { 1  if y > 0
       { 0  if y ≤ 0
```

### Why Step Function Only?

**The Limitation is the Point:**
- Step function is **not differentiable** at y=0
- This means we cannot use gradient descent
- This limitation is INTENTIONAL - it teaches why we need:
  - Continuous activations (sigmoid, ReLU) 
  - Gradient-based learning (backpropagation)
  - Multi-layer networks (MLP)

**Pedagogical Decision:**
Using sigmoid with perceptron learning rule would be conceptually wrong - that's actually the "Delta Rule" (a different algorithm). We keep concepts pure: step function + perceptron rule here, then sigmoid + gradient descent in MLP.

**Where You'll See Sigmoid:**
In the MLP implementation (next!), where it makes theoretical sense with gradient descent and backpropagation.