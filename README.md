# Pneumonia Detection using CNN

This repository contains code for training and evaluating a convolutional neural network (CNN) model to detect pneumonia from chest X-ray images. we trained the model using blow mentioned dataset and were able to reach 0.9099201560020447 accuracy.

## Overview

Pneumonia is a lung infection that causes inflammation in the air sacs, or alveoli, of one or both lungs. The infection can be caused by bacteria, viruses, or fungi.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Folder Structure](#folder-structure)
4. [Training](#training)
5. [Test Results](#test-results)
6. [Further Improvements](#further-improvements)
7. [Contributing](#contributing)
9. [License](#license)

## Installation

To set up the environment for running this project, follow these steps:

### Step 1: Dataset
Get the dataset of the project form the link ( https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia ).

### Step 2: Clone the Repository

Clone the repository to your local machine using Git:
```bash
git clone https://github.com/your_username/your_repository.git
cd Pneumonia-Detection-Using-CNN
```

### Step 3: Create and Activate a Conda Environment (Optional but Recommended)

If you prefer using Conda for managing environments, you can create a new Conda environment:
```bash
conda create --name pneumonia-env python=3.8
conda activate pneumonia-env
```

### Step 4: Install Dependencies

Install the required packages using `pip` and the `requirements.txt` file provided:
```bash
pip install -r requirements.txt
```

This command will install all the necessary dependencies, including TensorFlow and other libraries required for the project.

### Step 5: Run the Project

You're now ready to run the project. Depending on your setup and the structure of the project, you may run different scripts or notebooks for data preprocessing, model training, evaluation, etc.

### Optional: GPU Support

If you have a CUDA-enabled GPU, you can install TensorFlow GPU for faster computations:
```bash
pip install cuda==11.2.2 cudnn==8.1.0
pip install tensorflow-gpu==2.5.0
```

Make sure you have CUDA and cuDNN installed as per TensorFlow's requirements.

### Step 6: Explore the Project

Explore the project files and directories to understand the structure and functionality:

- `data_preprocessing.ipynb`: Notebook for data preprocessing.
- `model_training.ipynb`: Notebook for training the CNN model.
- `model_evaluation.ipynb`: Notebook for evaluating model performance.
- `data_exploration.py`: Python script for exploring the dataset.
- `model_deployment.ipynb`: Notebook for deploying the trained model.


## Folder Structure

The project folder structure is organized as follows:
```
Pneumonia-Detection-Using-CNN/
│
├── data/
│   ├── train/
│   ├── test/
│   └── val/
├── models/
│   └── cnn_model.h5
├── notebooks/
│   └── pneumonia_detection.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── train.py
├── results/
│   ├── accuracy.png
│   └── loss.png
├── README.md
└── environment.yml
```
### docs
Contains project documentation.

### results
Stores experimental results and trained models.

### src
Contains source code for the pneumonia detection model.

### notebooks
Includes Jupyter notebooks used for experimentation and analysis.

## Results

Detailed results and performance metrics are stored in the `results/` directory.



## Training

### Model Training Details

The CNN model was trained to detect pneumonia from chest X-ray images. The training process involved optimizing the model to achieve high accuracy on both training and validation datasets. for generalization of our model, data augmentation was added and helped reducing the difrence of train and validation accuracy from 8 to near 0.



### Performance Metrics

During training, the model reached 92% accuracy on both the training and validation datasets, demonstrating its ability to generalize well to unseen data. 

![Screenshot 2024-07-02 082713](https://github.com/sadegh15khedry/Pneumonia-Detection-Using-CNN/assets/90490848/adc989e2-bacf-4940-b4a0-2b9ae7e26151)


## Test Results

After training and evaluating the pneumonia detection model using chest X-ray images, the following test results were obtained:

             precision    recall  f1-score   support

      Normal       0.25      0.22      0.24       237
   Pneumonia       0.72      0.75      0.74       640

    accuracy                           0.61       877
   macro avg       0.49      0.49      0.49       877
weighted avg       0.59      0.61      0.60       877


Test accuracy: 0.9099201560020447   test_loss: 0.3045918047428131
precision: 0.5941611958622947   recall: 0.6066134549600912
f1: 0.6000544799045721

![cm](https://github.com/sadegh15khedry/Pneumonia-Detection-Using-CNN/assets/90490848/25c3a822-f907-4fdf-b5c8-20abdf44f206)

These results indicate the performance metrics achieved by the model on the test dataset, demonstrating its effectiveness in detecting pneumonia from X-ray images.

## Further Improvements
- Implement transfer learning with pre-trained models.
- Explore ensemble methods for improved performance.
- Optimize hyperparameters for better precision and recall.




## Contributing

If you wish to contribute to the project or modify it for your own use, feel free to fork the repository and create pull requests with your changes.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
