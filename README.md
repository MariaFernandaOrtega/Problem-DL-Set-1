## **Neural Network Implementation**

In this project, you will implement several versions of a fully connected neural network for image classification tasks. Specifically, you will build:

1. A neural network **from scratch** using only basic Python and NumPy.
2. A neural network using **PyTorch**.
3. A **TensorFlow**-based neural network, which has been partially implemented and will require debugging to work correctly.

By exploring these different implementations, you’ll gain insight into how deep learning libraries like PyTorch and TensorFlow abstract away lower-level details, allowing for faster and easier model development.

Each approach is provided with a template file that contains the necessary structure. Your job is to complete the missing methods. Please avoid modifying any other parts of the template to ensure consistent evaluation.

All three models will tackle the same classification problem, allowing you to verify your implementation and compare performance across different frameworks.

### **Dataset**

You will use the **MNIST** dataset of handwritten digits—a standard benchmark in image classification tasks. The MNIST dataset contains grayscale images of digits from 0 to 9, with each image having a resolution of **28x28 pixels**. In tensor form, each image is represented as a **28x28x1** array.

The task is to classify which digit (0 through 9) appears in each image. This is a **multiclass classification** problem with **k = 10** output classes.

### **Model Requirements**

Your neural networks should meet the following criteria:

* Allow for an **arbitrary number of hidden layers**, with a customizable number of neurons in each layer.
* Use the **sigmoid activation function** in all hidden layers.
* Use the **softmax activation function** in the output layer.
* Use the **cross-entropy loss function** for training.

> **Note:** In the from-scratch implementation (Part a), you will use **mean squared error (MSE)** as the loss function instead of cross-entropy. While MSE is not ideal for classification tasks, it is chosen here for simplicity. In practice, always prefer cross-entropy or log loss for classification problems.

---

Let me know if you want a summary or breakdown of any specific part of the project!

