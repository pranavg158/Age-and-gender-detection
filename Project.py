from flask import Flask, render_template, request, send_from_directory
import cv2
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16

base_model1 = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
for layer in base_model1.layers:
    layer.trainable = False
from tensorflow.keras import layers
# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model1.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model1 = tf.keras.models.Model(base_model1.input, x)

model1.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

model1.load_weights('modelg_weights.h5')






from tensorflow.keras.applications.vgg16 import VGG16

base_model2 = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
for layer in base_model2.layers:
    layer.trainable = False
from tensorflow.keras import layers
# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model2.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(8, activation='softmax')(x)

model2 = tf.keras.models.Model(base_model2.input, x)

model2.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'sparse_categorical_crossentropy',metrics = ['acc'])

model2.load_weights('modela_weights.h5')






COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction1 = model1.predict(img_arr)
    prediction2 = model2.predict(img_arr)
    prediction2 = np.argmax(prediction2)

    COUNT += 1
    print(prediction2)
    return render_template('prediction.html', data1=prediction1, data2=prediction2)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)
