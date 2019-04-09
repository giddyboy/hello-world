import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from keras.utils import to_categorical
from keras.utils import np_utils
import pickle

# Categorical: [0,1,2] -> 1 is now [0 1 0] 2 is now [0 0 1] 0 is now [1 0 0]
fileNumber = 100000
IMG_SIZE = 64
result2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(result2)
print(result2[1])
result2_encoded = to_categorical(result2)
print(result2_encoded[1])
training_data = []
oldX = []

yr=[]
def create_training_data():
    global yr
    path = "/home/philip/PycharmProjects/Chess01/ChessDone"
    pathR = "/home/philip/PycharmProjects/Chess01/ChessDoneResults"
    print(path)

    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # necessary since not greyscale
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        old_x = cv2.resize(img_array, (512, 512))
        # Read results
        resultimage = "y" + img
        resultimage = resultimage.lower()
        fileR = open(pathR + "/"+resultimage, "r", encoding='utf-8-sig')

        for line in fileR:
            a = line
            yr.append(int(a))

        # y_result = fileR.readlines()
        # for i in range(len(y_result)):
        #     x = y_result[i].replace('\n', "")
        #     y_result[i]=x
        #resultEncode = np_utils.to_categorical(y_result, num_classes = 13)
        training_data.append([new_array, yr])
        oldX.append([old_x])

        yr = []
        fileR.close()




create_training_data()

print(yr)


import random
random.seed(4)
random.shuffle(training_data)
random.seed(4)
random.shuffle(oldX)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # needed for later
y = np.array(y).reshape(-1,64)

oldX = np.array(oldX).reshape(-1,512,512,1)

for m in range(oldX.shape[0]):
    cv2.imwrite("oldX" + str(100000+m) + ".png", oldX[m])

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


pickle_out = open("oldX.pickle", "wb")
pickle.dump(oldX, pickle_out)
pickle_out.close()

print(X.shape)
print(y.shape)
print(y[0].T)
