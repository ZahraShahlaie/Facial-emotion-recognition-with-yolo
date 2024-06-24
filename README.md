# Facial Emotion Recognition Project 

### Overview

This project aims to develop a facial emotion recognition system using YOLOv8 for object detection and deep learning techniques for emotion classification. The dataset used for this project is sourced from Kaggle and consists of images labeled with various emotions. The process involves data preprocessing, handling imbalanced data, training, and validating a model to recognize emotions from facial images.

### Key Steps

1. **Environment Setup and Dependency Installation**
    - Various Python libraries are installed to set up the environment for the project. These include `ultralytics`, `torch_utils`, `deep_sort_realtime`, and others necessary for handling data and building models.

2. **Cloning YOLOv8-Face Repository**
    - The `yolov8-face` repository is cloned to utilize pre-trained models and scripts for face detection.

3. **Downloading and Preparing the Dataset**
    - The Kaggle API is used to download the "face-expression-recognition-dataset".
    - The dataset is extracted and prepared for further processing.

4. **Preprocessing the Dataset**
    - Data is organized into training and validation sets.
    - The dataset is cleaned by removing unnecessary folders and restructuring the data directories.

5. **Handling Imbalanced Data**
    - Imbalance in the dataset is addressed by using the `RandomOverSampler` from `imblearn` to ensure that each emotion class has an adequate number of samples for training and validation.

6. **Visualization**
    - Bar plots are created using `seaborn` and `matplotlib` to visualize the distribution of images across different emotion classes for both training and validation datasets.

7. **Creating TensorFlow Datasets**
    - Convert the resampled datasets into TensorFlow datasets for training, validation, and testing.

8. **Defining Constants and Data Augmentation**
    - Define constants like batch size and image size.
    - Set up data augmentation strategies.

9. **Model Definition and Training**
    - Define and compile the convolutional neural network (CNN) model.
    - Train the model using the training and validation datasets.

10. **Model Evaluation**
    - Evaluate the model's performance on the test dataset.
    - Generate and display a confusion matrix and classification report.

11. **Face Detection and Emotion Recognition**
    - Use a pre-trained YOLOv8 model to detect faces in images.
    - Crop detected faces and classify their emotions using the trained CNN model.

### Detailed Steps

#### 1. Environment Setup

Install the necessary libraries using `pip`.

#### 2. Cloning the YOLOv8-Face Repository

Clone the YOLOv8-face repository.

#### 3. Downloading and Preparing the Dataset

Download the dataset using Kaggle API.

Extract the dataset.

#### 4. Preprocessing the Dataset

Remove unnecessary folders and restructure directories.

#### 5. Handling Imbalanced Data

Calculate the number of images in each class for the training set and visualize.

Oversample the training set to balance the data.

Repeat the same steps for the validation set.

#### 6. Visualization

Create bar plots to visualize the distribution of images across different emotion classes for both training and validation datasets.

#### 7. Creating TensorFlow Datasets

1. **Label Encoding**
    - Encode the labels using `LabelEncoder`.

2. **Splitting Data**
    - Split a portion of the training data for testing.

3. **Creating TensorFlow Datasets**
    - Create TensorFlow datasets for training, validation, and testing.

4. **Configuring Datasets for Performance**
    - Configure datasets using caching, shuffling, and prefetching for optimal performance.

5. **Printing Dataset Sizes**
    - Print the sizes of the training, validation, and testing datasets.

6. **Class Distribution**
    - Print the count of samples in each class for training, validation, and testing datasets.

7. **Showing Labels**
    - Display the unique class names and their count.

#### 8. Defining Constants and Data Augmentation

1. **Define Constants**
    - Set the batch size and image size.

2. **Data Augmentation**
    - Set up a data augmentation pipeline using `keras.Sequential`.

#### 9. Model Definition and Training

1. **Define the Model**
    - Create a CNN model using `Sequential` from `keras`.

2. **Compile the Model**
    - Compile the model with the Adam optimizer and sparse categorical crossentropy loss.

3. **Train the Model**
    - Train the model using the training and validation datasets, with early stopping as a callback.

#### 10. Model Evaluation

1. **Evaluate the Model**
    - Evaluate the model's performance on the test dataset and print the loss and accuracy.

2. **Generate Confusion Matrix**
    - Create and display a confusion matrix for the test dataset predictions.

3. **Generate Classification Report**
    - Print a classification report detailing the precision, recall, and F1-score for each class.

#### 11. Face Detection and Emotion Recognition

1. **Load Pre-trained YOLOv8 Model**
    - Download the pre-trained YOLOv8 model from [yolov8n-face.pt](https://github.com/akanametov/yolov8-face/tree/dev) and upload it to `/content`.

2. **Define Functions**
    - **Face Detection**: Detect faces in the input image using the YOLOv8 model.
    - **Crop Faces**: Crop the detected faces from the input image.
    - **Classify Emotions**: Use the trained CNN model to classify the emotions of the cropped faces.
    - **Annotate Images**: Annotate the input image with detected faces and their corresponding emotions.

3. **Helper Function**
    - **Pascal VOC to COCO Format**: Convert bounding box coordinates from Pascal VOC format to COCO format.

### Summary

This project guide outlines the process of setting up a facial emotion recognition system using a YOLOv8-based face detection model and a deep learning model for emotion classification. The key steps involve setting up the environment, downloading and preparing the dataset, preprocessing, handling imbalanced data, creating TensorFlow datasets, defining and training a CNN model, evaluating the model's performance, and integrating face detection and emotion recognition. By following these steps, you will be able to build and deploy a robust system capable of detecting faces in images or videos and accurately classifying their emotions.

## Further Enhancements

1. **Model Optimization**: Further optimize the model by experimenting with different architectures, hyperparameters, and training techniques.
   
2. **Real-time Emotion Recognition**: Extend the system to work in real-time with live video feeds for applications in surveillance, human-computer interaction, etc.
   
3. **Dataset Expansion**: Expand the dataset to include more diverse faces and emotions to improve generalization.
It seems like you have provided code snippets for several functions and processes related to facial emotion recognition and webcam interaction in Python, particularly with OpenCV and TensorFlow. Here’s a brief overview and summary of the functionalities based on the provided code:

1. **Facial Emotion Recognition Project Overview**:
   - You are using a combination of YOLOv8 for face detection and a CNN model for emotion classification.
   - The project involves setting up the environment, downloading datasets, preprocessing, handling imbalanced data, training the CNN model, evaluating its performance, and integrating it with face detection using YOLOv8.

2. **Key Components and Functions**:
   - **Environment Setup**: Python libraries (`ultralytics`, `torch_utils`, etc.) are installed for dependencies.
   - **Dataset Handling**: Kaggle API is used to download the "face-expression-recognition-dataset".
   - **Data Preprocessing**: Includes organizing, cleaning, and restructuring the dataset directories.
   - **Handling Imbalanced Data**: Using `RandomOverSampler` from `imblearn` to balance data classes.
   - **Model Training and Evaluation**: CNN model defined using TensorFlow/Keras, trained on the dataset, and evaluated using metrics like accuracy and confusion matrix.
   - **Face Detection and Emotion Recognition**: Integrating YOLOv8 for face detection in images and videos, followed by emotion classification using the trained CNN model.
   - **Webcam Interaction**: Functions for capturing images and streaming video from a webcam, with real-time face detection and emotion recognition.

3. **Integration with Webcam**:
   - **Webcam Image Capture**: Using JavaScript to capture images from the webcam, converting them into OpenCV images, detecting faces, and recognizing emotions.
   - **Webcam Video Streaming**: JavaScript functions enable real-time video streaming from the webcam, with overlaid bounding boxes indicating detected faces and their emotions.

4. **Output Display**:
   - Functions to display processed images and videos, including embedding videos directly into HTML using Base64 encoding.

5. **Improvements and Future Work**:
   - Ensure robustness and efficiency of face detection and emotion recognition algorithms in real-time scenarios.
   - Consider deploying the system in environments requiring continuous monitoring of emotions (e.g., customer sentiment analysis, educational settings).
   - Enhance user interaction and feedback mechanisms for better usability and performance.


### Contribution
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

### Contact
For any inquiries or support, please contact [za.shahlaie@gmail.com](mailto:za.shahlaie@gmail.com).
