# Custom ResNet Model for CIFAR10

This repository contains a custom ResNet model implementation for the CIFAR10 dataset. The goal of this project is to achieve a classification accuracy of 90% or higher on the CIFAR10 dataset, using techniques such as Albumentations for image augmentation and dropout to reduce overfitting.

## ResNet (Residual Network)

The main idea behind ResNet is to address the degradation problem that occurs when deeper neural networks are trained. It has been observed that as the network depth increases, the accuracy of the network starts to saturate and then degrade rapidly. The degradation problem arises due to the optimization difficulties in training deep networks and the vanishing/exploding gradient problem.

ResNet introduces the concept of residual learning to overcome the degradation problem. In a ResNet, instead of directly learning the underlying mapping, the network learns the residual mapping. This is achieved by introducing skip connections or "shortcut connections" that bypass one or more layers. By propagating the identity mapping through the shortcut connections, the gradient can flow directly from the beginning to the end of the network, enabling the learning of deeper networks.


## Skip Connections and their Implementation:

In the context of deep learning and ResNet, skip connections, also known as shortcut connections or identity mappings, refer to the direct connections that bypass one or more layers in a neural network. These connections allow the gradient to flow more easily during backpropagation and help address the degradation problem that arises when training deep networks.

In this repository, skip connections are implemented using a custom layer called `CustomLayer` present in 'resnet.py'. The `CustomLayer` consists of a convolutional layer and a residual layer.

During the forward pass:
1. The input is first passed through the convolutional layer.
2. If a residual layer is present (controlled by the `res` parameter), the input is saved as a reference.
3. The output of the residual layer is added to the saved input, enabling the gradient to flow directly from the input to the output.
4. The resulting output is then passed to the next layer in the network.

The skip connections in `CustomLayer` allow the model to learn residual mappings, which capture the difference between the input and the desired output. By adding the residual to the input, the model can easily learn to make small adjustments and focus on learning the residual information.

## Repository Structure

The repository is structured as follows:




