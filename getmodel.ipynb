import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Convolution2D
import numpy as np
import _pickle
import cv2
import sys
from copy import copy, deepcopy

#import matplotlib.pyplot as plt
import tkinter

from tkinter import *
import PIL.Image, PIL.ImageTk     #from PIL import Image, ImageTk
#import cv2
from multiprocessing import Pool, cpu_count
import multiprocessing
#pool= multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 5))
#pool = Pool(max(cpu_count()//2 - 2, 1))
#pool= multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 5))

pickle_in = open("X.pickle", "rb")
X = _pickle.load(pickle_in)

pickle_in2 = open("y.pickle", "rb")
y = _pickle.load(pickle_in2)
#X = X.reshape(-1,1)
print(X.shape)
print(y.shape)

pickle_in2 = open("oldX.pickle", "rb")
oldX = _pickle.load(pickle_in2)

# cv2.imshow('image', X[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#oldX = deepcopy(X)
print(y.shape)
X = X / 255.0
y_old = deepcopy(y)
y = y / 12.0
# X2 = np.zeros([27,100,100,1])
# X3 = np.zeros([27,64,64,1])
# y2 = np.zeros([27, 3])

# print(X2.shape[1:])
# print(y2)

model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=X.shape[1:]))
model.add(Activation('relu'))


model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(64))
#model.add(Dense(256, input_shape=X2.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vector



# model.add(Conv2D(256, (2, 2), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# #
# model.add(Conv2D(256, (2, 2)))
# model.add(Activation('relu'))
# #
#
# model.add(Conv2D(256, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(256, (2, 2)))
# model.add(Activation('relu'))
# #model.add(MaxPooling2D(pool_size=(2, 2)))
# #
# #
# model.add(Conv2D(256, (2, 2)))
# model.add(Activation('relu'))
#
# model.add(Conv2D(256, (2, 2)))
# model.add(Activation('relu'))
#

# model.add(MaxPooling2D(pool_size=(2, 2)))
#


model.add(Dense(64)) #, input_shape=X.shape[1:]))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))
#
# model.add(Dense(256))
# model.add(Activation('relu'))
#
# model.add(Dense(256))
# model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('sigmoid'))
print (model.output_shape)
#model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, y, epochs=600, validation_split=0.1)
test0=model.predict(X)
test0 = test0*12
test0 = np.around(test0)

success = 0
for tem in range(X.shape[0]):
    if (test0[tem]==y_old[tem]).all():
        success = success+1

print(str(success) + " out of " + str(X.shape[0]))


for m in range(20):
    position = int(input("Enter position: "))
    if position == -1:
        sys.exit()
    else:
        #print(test0[position])



        from PIL import Image, ImageDraw, ImageFont

        # cimage = cStringIO.StringIO(cimage.getvalue())
        # cimage = Image.open(cimage)

        im = Image.new('RGBA', (512, 512), (0, 0, 0, 0)) # Create a blank image
        draw = ImageDraw.Draw(im) # Create a draw object
        colors = 'green'
        n = 0
        for j in range(9):
            for i in range(9):

                if n == 0:
                    colors = 'red'
                    n = 1
                elif n == 1:
                    colors = 'green'
                    n = 0
                draw.rectangle((i * 64, j * 64, i*64+64, j*64 + 64), fill=colors, outline=colors)
            i = 0
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 50)
        for k in range(64):
            if test0[position][k] == 1:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'K',font=font, fill = "black")
            if test0[position][k] == 2:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'K',font=font, fill = "white")
            if test0[position][k] == 3:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'Q',font=font, fill = "black")
            if test0[position][k] == 4:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'Q',font=font, fill = "white")
            if test0[position][k] == 5:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'P',font=font, fill = "black")
            if test0[position][k] == 6:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'P',font=font, fill = "white")
            if test0[position][k] == 7:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'R',font=font, fill = "black")
            if test0[position][k] == 8:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'R',font=font, fill = "white")
            if test0[position][k] == 9:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'N',font=font, fill = "black")
            if test0[position][k] == 10:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'N',font=font, fill = "white")
            if test0[position][k] == 11:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'B',font=font, fill = "black")
            if test0[position][k] == 12:
                yc = k // 8
                xc = k - 8 * yc
                draw.text((xc*64+15, yc*64 + 5), 'B',font=font, fill = "white")


        im.show()

#X = X * 255.0

    img = Image.open("oldX"+str(100000+position)+".png")
    img.show()


    # cv2.imshow('image', oldX[position])
    # cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()

# pred = model.predict(X)
#
# print("[0 1 2 3 4 5 6 7 8 9]")
#
# indices = np.argmax(pred[0])
# print(indices)
# counts = 0
# #---------------------------------------------------------------
# pickle_in = open("X.pickle", "rb")
# X = _pickle.load(pickle_in)
#
# pickle_in2 = open("y.pickle", "rb")
# y = _pickle.load(pickle_in2)
#
# X = X / 255.0
# X = X * 256
#
# current_position = 5
#
# window = tkinter.Tk()
# window.geometry("640x480")
#
# IMG_SIZE = 64
#
#
# X16 = X[current_position]
# X16 = cv2.resize(X16, (IMG_SIZE, IMG_SIZE))
#
#
# print(X16.shape)
