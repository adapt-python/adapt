"""
Utility functions for adapt package.
"""

import tarfile
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_tar_file(path):
    if path.endswith("tar.gz"):
        tar = tarfile.open(path, "r:gz")
        tar.extractall()
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(path, "r:")
        tar.extractall()
        tar.close()


def load_mnistm(path="mnist_m/"):
    from skimage import transform
    from skimage.color import rgb2gray
    
    image_files = np.sort(os.listdir(path+'mnist_m_train/'))
    x_mnist_m_train = []
    #Labels
    df = pd.read_csv(path+'mnist_m_train_labels.txt', sep=' ', header = None)
    y_mnist_m_train = list(df[1].values)
    i = 0
    print("Train...")
    for img in image_files:
        image_path = path+'mnist_m_train/'+img
        img = plt.imread(image_path)
        img = rgb2gray(img)
        img = transform.resize(img, (28, 28), mode='symmetric')
        x_mnist_m_train.append(img)
        i += 1
        if i % 1000 == 0:
            print("Processed images: %i"%i)

    image_files = np.sort(os.listdir(path+'mnist_m_test/'))
    x_mnist_m_test = []
    #Labels
    df = pd.read_csv(path+'mnist_m_test_labels.txt', sep=' ', header = None)
    y_mnist_m_test = list(df[1].values)
    i = 0
    print("Test...")
    for img in image_files:
        image_path = path+'mnist_m_test/'+img
        img = plt.imread(image_path)
        img = rgb2gray(img)
        img = transform.resize(img, (28, 28), mode='symmetric')
        x_mnist_m_test.append(img)
        i += 1
        if i % 1000 == 0:
            print("Processed images: %i"%i)
    
    X_train = np.stack(x_mnist_m_train, 0)    
    X_test = np.stack(x_mnist_m_test, 0)
    
    y_train = np.array(y_mnist_m_train)
    y_test = np.array(y_mnist_m_test)
    return ((X_train, y_train), (X_test, y_test))



def load_office(path="", domain="webcam"):
    from skimage import transform

    dir_list = np.sort(os.listdir(path+domain+'/images/'))

    X = []
    y = []
    for dir_ in dir_list:
        print("Extract %s"%dir_)
        image_list = np.sort(os.listdir(path+domain+'/images/'+dir_+"/"))
        for img in image_list:
            img = plt.imread(path+domain+'/images/'+dir_+"/"+img)
            img = transform.resize(img, (224, 224, 3), mode='symmetric')
            img = (img * 255).astype(np.uint8)
            X.append(img)
            y.append(dir_)
    return X, y



# def rgb2gray(rgb):
#     return rgb.dot(np.array([0.2989, 0.5870, 0.1140]))

# def crop(img, shape=(28, 28)):
#     old_shape = img.shape[1:]
#     assert old_shape[0] >= shape[0]
#     assert old_shape[1] >= shape[1]
#     x_pad = int((old_shape[0]-shape[0])/2)
#     y_pad = int((old_shape[1]-shape[1])/2)
#     return img[:, x_pad:x_pad+shape[0], y_pad:y_pad+shape[1]]