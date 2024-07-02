from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, test_dataset):
     # Get the true labels
    true_labels = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
    true_labels = np.argmax(true_labels, axis=1)  # Convert one-hot to class indices
    
    # Make predictions
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    
    test_loss, test_acc = model.evaluate(test_dataset)
    cm = confusion_matrix(true_labels, predicted_labels)
    # predictions = model.predict(x_test)
    # report = classification_report(y_test, predictions)
    # cm = confusion_matrix(y_test, predictions)
    return test_loss, test_acc, cm


def get_test_dataset(test_dir, image_width, image_height):
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(image_height, image_width),
    color_mode='grayscale',
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    return test_dataset

def display_cofiution_matrix(cm, test_dataset):
    # Display the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Saving the plots
    plt.savefig('../results/training_validation_loss_and_accuracy.png')
    
    plt.show()
