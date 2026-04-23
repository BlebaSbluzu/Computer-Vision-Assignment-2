from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D, Conv2D, MaxPooling2D, Rescaling, BatchNormalization,RandomFlip, RandomRotation, RandomZoom
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import keras_tuner as kt

batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again
train_dir = '/home/bleba/Computer-Vision-Assignment-2/Training/train'
test_dir = '/home/bleba/Computer-Vision-Assignment-2/Training/test'


print("\nTRAIN DISTRIBUTION")
for cls in ["BACTERIAL", "NORMAL", "VIRAL"]:
    folder = os.path.join(train_dir, cls)
    count = len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])
    print(cls, count)


with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ',class_names)
    num_classes = len(class_names)
    
    class_weight = {
    0: 0.70,   # bacterial
    1: 1.24,   # normal
    2: 1.33    # viral
    }

    print("Class weights:", class_weight)


    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()


    def model_builder(hp):
        model = tf.keras.models.Sequential([

        RandomFlip("horizontal"),
        RandomRotation(0.05),
        RandomZoom(0.1),
        
        Rescaling(1.0/255),
        Conv2D(hp.Int("conv1_filters", min_value=16, max_value=64, step=16), (3,3), activation = 'relu', input_shape = (img_height,img_width, img_channels)),
        MaxPooling2D(2,2),
        Conv2D(hp.Int("conv2_filters", min_value=32, max_value=128, step=32), (3,3), activation = 'relu'),
        MaxPooling2D(2,2),
        Conv2D(hp.Int("conv3_filters", min_value=32, max_value=128, step=32), (3,3), activation = 'relu'),
        MaxPooling2D(2,2),
        # Flatten(), # flatten multidimensional outputs into single dimension for input to dense fully connected layers
        # Dense(512, activation = 'relu'),
        # Dropout(0.2),

        #Helps with overfitting instead cuz it reduces the number of trainable
        # parameters and prevents the model from memorising the training data.
        GlobalAveragePooling2D(),
        Dropout(hp.Choice("dropout_rate", values=[0.2, 0.3, 0.4, 0.5])),


        Dense(num_classes, activation = 'softmax')
    ])

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])

        return model


    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory="keras_tuner_dir",
        project_name="pneumonia_tuning"
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weight
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best conv1 filters:", best_hps.get("conv1_filters"))
    print("Best conv2 filters:", best_hps.get("conv2_filters"))
    print("Best conv3 filters:", best_hps.get("conv3_filters"))
    print("Best dropout:", best_hps.get("dropout_rate"))
    print("Best learning rate:", best_hps.get("learning_rate"))


    model = tuner.hypermodel.build(best_hps)
    
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras",save_freq='epoch',save_best_only=True)

    if fit:
        history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[save_callback],
        epochs=epochs,
        class_weight=class_weight
        )
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    #if shuffle=True when creating the dataset, samples will be chosen randomly   
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    
    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(),0))
            plt.title('Actual:' + class_names[labels[i].numpy()]+ '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()