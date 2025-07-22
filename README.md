

## **Neural Network Implementation for MNIST Classification**

In this project, you will implement and evaluate fully connected neural networks for image classification, using multiple approaches:

1. A neural network implemented **from scratch** using only Python and NumPy.
2. A model built with **PyTorch**.
3. A partially completed neural network in **TensorFlow/Keras** that you will debug and enhance.

By comparing these approaches, you will learn how modern libraries simplify the construction of deep learning models by abstracting low-level operations, while also gaining insight into the mechanics of neural networks at a foundational level.

---

### **Dataset: MNIST Handwritten Digits**

You will work with the **MNIST dataset**, a widely used benchmark in computer vision. It consists of grayscale images of handwritten digits from 0 to 9, each with a resolution of **28×28 pixels**.

**Figure 1:** Example images from MNIST dataset with 28×28 pixel resolution.

The training dataset contains **n** labeled examples of the form:

$$
(x^{(i)}, y^{(i)}), \quad i = 1, \dots, n
$$

where:

* $x^{(i)}$ is a 28×28 grayscale image.
* $y^{(i)} \in \{0,1\}^k$ is the **one-hot encoded** class label with $k = 10$ (digits 0–9). The j-th element $y^{(i)}_j = 1$ if the image belongs to class j, and 0 otherwise.

Your neural network will output a probability vector $\hat{y} = h_\theta(x) \in \mathbb{R}^k$ using the **softmax activation** function. The element $\hat{y}_j$ indicates the model's confidence that the input belongs to class j. For prediction, the final class label corresponds to the index with the highest score in $\hat{y}$.

---

### **Neural Network Design Requirements**

* **Architecture**: Arbitrary number of hidden layers, with configurable neurons per layer.
* **Activation Functions**:

  * **Sigmoid** for all hidden layers.
  * **Softmax** for the output layer.
* **Loss Functions**:

  * Use **cross-entropy** for PyTorch and TensorFlow implementations.
  * Use **mean squared error (MSE)** in the scratch implementation (for simplicity). *Note: MSE is not ideal for classification but is acceptable here for educational purposes.*

---

### **Tasks**

#### **1. Complete the Network Implementations**

You are provided with three Python templates:

**(a) `network_scratch.py`** — Neural network implemented from scratch
**(b) `network_pytorch.py`** — Neural network implemented using PyTorch

> Templates contain `TODO` comments indicating where code must be completed.

A GitHub Classroom link will generate a private repository for your team:
[GitHub Assignment Link](https://classroom.github.com/a/XgELWE_C)

#### **2. Fix and Extend the TensorFlow Implementation**

**`network_tensorflow.py`** contains a partially implemented neural network using **TensorFlow and Keras**. Your tasks:

* **(a)** Identify and **fix three bugs** in the code that prevent the model from running correctly.
* **(b)** Implement a custom **time-based learning rate scheduler**:

```python
class TimeBasedLearningRate:
    def __init__(self, initial_rate: int):
        # Initializes with a positive integer learning rate
        # Decreases by 1 at each step until it reaches 1
```

#### **3. Evaluate Network Performance**

In the provided Jupyter notebook `MNIST_classification.ipynb`:

* **(a)** Plot the **training and validation accuracy** as a function of **epochs** for all three implementations.
* **(b)** Ensure the notebook **prints training progress**, showing accuracy/loss per epoch.
* **(c)** At the **top of the notebook**, include:

  * Team number
  * Team member names
  * Matriculation numbers (as comments)

---

### **Notes on Evaluation**

* Perfect accuracy is **not expected**, especially with limited training data.
* Your goal is to demonstrate **learning behavior**:

  * Accuracy should **increase over epochs**.
  * Final accuracy should be **clearly better than random guessing** (i.e., \~10%).





