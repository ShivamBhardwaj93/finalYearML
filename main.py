from google.colab import drive
drive.mount('/content/drive')

pip install tensorflow==2.16.1 keras==3.3.3 tensorboard==2.16.2

 !unzip '/content/drive/MyDrive/Colab Notebooks/dataset.zip' -d './content/'

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import numpy as np


IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50

path = "/content/content/dataset"

train_ds , test_ds = keras.utils.image_dataset_from_directory(
    path ,
    image_size=(256,256),
    batch_size=128 ,
    shuffle=True,
    seed = 123 ,
    validation_split=.2,
    subset='both'
)


class_names = train_ds.class_names
class_names

# Initialize MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = keras.Sequential([
    keras.layers.Rescaling(scale = 1/255 , input_shape =(256,256,3) ) ,

    keras.layers.Conv2D(32 , (3,3) , activation = 'relu'),
    keras.layers.MaxPool2D((2,2))                     ,
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64 , (3,3) , activation = 'relu') ,
    keras.layers.MaxPool2D((2,2)) ,
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64 , (3,3) , activation = 'relu') ,
    keras.layers.MaxPool2D((2,2)) ,
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64 , (3,3) , activation = 'relu') ,
    keras.layers.MaxPool2D((2,2)) ,
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(128 , (3,3) , activation = 'relu') ,
    keras.layers.MaxPool2D((2,2)) ,

    # fully connected layers

    keras.layers.Flatten(),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.Dense(38,activation ='sigmoid')

  ])

  model.compile(
    optimizer = 'adam' ,
    loss = 'sparse_categorical_crossentropy',
    metrics = 'accuracy'
  )


model.summary()
history = model.fit(train_ds , epochs = 15)



accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(1,16)

plt.plot(epochs , accuracy , label = 'Acuuracy')
plt.plot(epochs , loss , label = 'loss')
plt.legend()
plt.show()

model.evaluate(test_ds)

def img_to_pred(image):
  image = image.numpy()
  image = tf.expand_dims(image,0)
  return image

model.save('/content/drive/MyDrive/model.keras')

plt.figure(figsize=(18,18))
for images, labels in test_ds.take(1) : # take the first patch
  for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(images[i].numpy().astype('uint32'))
    plt.axis('off')
    actual = class_names[labels[i]]
    predict =class_names[np.argmax( model.predict(img_to_pred(images[i])))]
    plt.title(f"actual : {actual}  \n predicted : {predict} ")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
import keras

# Make predictions
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = test.classes

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
cls_report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", cls_report)


model.evaluate(test_ds)

