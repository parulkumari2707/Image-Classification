Image Classification using Convolutional Neural Networks (CNNs)
===
This project implements an image classification model using Convolutional Neural Networks (CNNs) to classify images into predefined categories (e.g., animals, vehicles, fruits) using the CIFAR-10 dataset.

Description
===
The goal of this project is to demonstrate the use of CNNs for image classification tasks. The CIFAR-10 dataset is used as the example dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The project includes the following steps:

1. Loading the CIFAR-10 dataset.
2. Preprocessing the images by resizing and normalizing them.
3. Defining and training a CNN model using TensorFlow and Keras.
4. Evaluating the model's performance on the test set.
5. Saving the trained model as an HDF5 file for future use.

The project also includes visualization of the training/validation accuracy and loss curves to analyze the model's performance during training.

Dataset
===
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 test images.

Installation
====
First, install the required dependencies using pip:
```bash
pip install tensorflow keras
```

Usage:
===
1. Clone the repository:
```bash
git clone https://github.com/parulkumari2707/Image-Classification.git
cd image-classification-cnn
```

2. Run the image_classification_cnn.py script:
```bash
python image_classification_cnn.py
```
This script loads the CIFAR-10 dataset, preprocesses the images, defines and trains a CNN model, evaluates the model, and saves the trained model as an HDF5 file (Image-classification-cnn.h5). The training/validation accuracy and loss are also plotted.

3. To load the trained model for inference:
```bash 
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('Image-classification-cnn.h5')

# Perform inference on new images (not included in the CIFAR-10 dataset)
# Replace 'image_path' with the path to your image file
image = cv2.imread('image_path')

# Preprocess the image
image = preprocess_image(image)

# Make predictions
predictions = model.predict(image)
```

Model Architecture
===
The CNN model architecture consists of convolutional layers followed by max-pooling layers, followed by fully connected layers. The final layer uses softmax activation for multiclass classification.

Conclusion
===
In conclusion, this project demonstrates the effectiveness of Convolutional Neural Networks (CNNs) for image classification tasks. The trained CNN model achieves a satisfactory level of accuracy on the CIFAR-10 dataset, showcasing its capability to learn and generalize patterns from images. By following the steps outlined in this project, users can train their own CNN models for image classification tasks on custom datasets.
