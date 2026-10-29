AI-Based Image Enhancement System

About the Project

This project focuses on image resolution enhancement using a deep learning model based on a Super-Resolution Convolutional Neural Network (SRCNN). The model learns to reconstruct images from low-resolution inputs using high-resolution training images.

Technologies Used

Python

PyTorch

Torchvision

Pillow

NumPy

Matplotlib

Datasets

DIV2K: Used for training the model (800 high-resolution images).

Set5: Used for testing the model (5 benchmark images).

Model Architecture

The project uses a three-layer convolutional neural network with ReLU activation functions. The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer.

Project Workflow

Load and preprocess images.

Generate low-resolution inputs using downscaling and bicubic interpolation.

Train the SRCNN-style model.

Save the trained model parameters.

Test the model and compare its output with bicubic interpolation and the original image.
