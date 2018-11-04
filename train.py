import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#import matplotlib.pyplot as plt
#import itertools


train_path = "train/"
valid_path = "valid/"
test_path = "test/"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['cat','dog'], batch_size=1)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['cat','dog'], batch_size=1)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['cat','dog'], batch_size=1)

#model = Sequential([
#	Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
#	Flatten(),
#	Dense(2, activation='softmax'),
#	])
#
#model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit_generator(train_batches, steps_per_epoch=60, validation_data=valid_batches, validation_steps=16, epochs=1, verbose=2)
#
#predictions = model.predict_generator(test_batches, steps=16, verbose=0)
#
#print(predictions)
#
#for i in range(len(test_batches)):
#	test_img, test_label = next(test_batches)
#	for j in range(1):
#		prediction = predictions[(1*i)+j]
#		print("Original : ",['cat','dog'][int((test_label[j])[1])],"----> Predicted : ",['cat','dog'][int(prediction[1])])

print("------------------------------------------------------------------------------------------------------------------------------")

vgg_model = keras.applications.vgg16.VGG16()

model2 = Sequential()
for layer in vgg_model.layers:
	model2.add(layer)
model2.layers.pop()
for layer in model2.layers:
	layer.trainable = False
model2.add(Dense(2, activation='softmax'))

model2.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit_generator(train_batches, steps_per_epoch=60, epochs=5, verbose=2)

predictions = model2.predict_generator(test_batches, steps=16, verbose=0)

print(predictions)

for i in range(len(test_batches)):
        test_img, test_label = next(test_batches)
        for j in range(1):
                print("Original : ",['cat','dog'][int((test_label[j])[1])],"----> Predicted : ",['cat','dog'][list(predictions[i+j]).index(max(predictions[i+j]))])
