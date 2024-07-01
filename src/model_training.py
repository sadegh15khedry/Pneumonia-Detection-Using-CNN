import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

def get_untrained_custom_model(imgage_width, imgage_height, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model = Sequential()
    # Convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(imgage_width, imgage_height, 3), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.25))    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(imgage_width, imgage_height, 3), kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))    
    model.add(Flatten())
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.5))
    # Output layer
    model.add(Dense(units=2, activation='softmax'))
    
    
    model.compile(optimizer=optimizer, loss=loss , metrics=metrics)
    return model


def train_model(model, train_dataset, epochs, val_dataset):
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
    history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
    )
    return history

def get_train_dataset(train_dir, batch_size, image_width, image_height):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    return train_dataset

def get_val_dataset(val_dir, batch_size, image_width, image_height):
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    return val_dataset