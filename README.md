Neural network implementation
You will implement different classes representing a fully connected neural net-
work for image classification problems. There are two classes of neural networks:
one using only basic packages and one using PyTorch. In addition to that, a
third class of neural network has been implemented using Tensorflow and re-
quires some fixing in order to function correctly.
As you work through this problem, you will see how those machine learning
libraries abstract away implementation details allowing for fast and simple con-
struction of deep neural networks.
For each approach, a Python template is supplied that you will need to
complete with the missing methods. Feel free to play around with the rest,
but please do not change anything else for the submission. All approaches will
solve the same classification task, which will help you validate that your code
is working and that the network is training properly.
For this problem, you will work with the MNIST dataset of handwritten
digits, which has been widely used for training image classification models. You
will build models for a multiclass classification task, where the goal is to predict
what digit is written in an image (to be precise, this is a k-class classification
task where in this case k = 10). The MNIST dataset consists of black and white images of digits 
from 0 to 9 with a pixel resolution of 28x28. Therefore,
in a tensor representation the images have the shape 28x28x1. The goal is
to classify what digit is drawn on a picture using a neural network with the
following characteristics:
• an arbitrary amount of hidden layers, each with arbitrary amount of neu-
rons
• sigmoid activation function for all hidden layers
• softmax activation function for the output layer
• cross entropy loss function. *
* For the implementation from scratch (Part (a)) we use a mean squared
error (MSE) loss function. This is not recommended for a classification task,
but we use it to simplify the implementation. In the future please consider using
cross entropy / log loss instead.
