from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Rescaling, RandomFlip, RandomRotation, RandomZoom, Input
from keras.applications import MobileNetV2
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report


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

    


    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.05),
        RandomZoom(0.1)
    ])

    base_model = MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    inputs = Input(shape=(img_height, img_width, img_channels))
    x = data_augmentation(inputs)
    x = Rescaling(1.0/255)(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
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



# F1 scores and precision, recall stuff here !!! 
    # y_true = []
    # y_pred = []

    # for images, labels in test_ds:
    #     predictions = model.predict(images)
    #     y_true.extend(labels.numpy())
    #     y_pred.extend(np.argmax(predictions, axis=1))

    # print(classification_report(y_true, y_pred, target_names=class_names))
    
    
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