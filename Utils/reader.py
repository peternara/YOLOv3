import tensorflow as tf
import numpy as np
import cv2
from os.path import isfile, join
from os import listdir
from Utils import extract_labels as extract_labels


'''--------Here fnish the minibatch operation--------'''
def images(batch_size, path):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    batch_filenames = []
    num = len(filenames) // batch_size
    for i in range(num):
        batch_filename = filenames[i * batch_size: (i + 1) * batch_size]
        batch_filenames.append(batch_filename)
    return batch_filenames


def get_image(path, width, height): 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    return image


def get_labels(batch_size, path):
    labels_filenames = images(batch_size, path)
    return labels_filenames


'''--------Test images--------'''
if __name__ == '__main__':
    image = images(16, '/home/sherman/projects/data/VOCdevkit/VOC2007/JPEGImages')
    print("Each batch size: {}".format(len(image[0])))
    print("Total batch number: {}".format(len(image)))
