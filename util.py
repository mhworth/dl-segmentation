
import sys, os, re
import collections
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from cStringIO import StringIO
from scipy.misc import imresize


REAL_WIDTH = 2048
REAL_HEIGHT = 2048

WIDTH = REAL_WIDTH/2
HEIGHT = REAL_HEIGHT/2

ClinicalInformation = collections.namedtuple('ClinicialInformation', ["filename"])

# Some functions to load image
def load_image_file(filename):
    a = np.fromfile(filename, dtype=">i2")
    raw_image = a.reshape((REAL_WIDTH, REAL_HEIGHT))
    return imresize(raw_image, 0.5)

def load_image_content(content):
    a = np.fromstring(content, dtype=">i2")
    raw_image = a.reshape((REAL_WIDTH, REAL_HEIGHT))
    return imresize(raw_image, 0.5)

def load_image_from_bucket(bucket, key_name):
    key = bucket.get_key(key_name)
    image = load_image_content(key.get_contents_as_string())
    return image

def resample_image(image16):
    image16float = image16.astype(np.float32)
    image8 = image16float*256/image16.max()
    return image8.astype(np.uint8)

def load_truth_data(bucket):
    nodule_truth = bucket.get_key('Clinical_Information/nodule-truth-cleaned.csv')
    non_nodule_truth = bucket.get_key('Clinical_Information/non-nodule-truth-cleaned.csv')
    
    nodule_truth_data, non_nodule_truth_data = nodule_truth.get_contents_as_string(), non_nodule_truth.get_contents_as_string()
    sio = StringIO(nodule_truth_data)
    sio.seek(0)
    ntd = pd.read_csv(sio, sep=',')
    
    sio = StringIO(non_nodule_truth_data)
    sio.seek(0)
    nntd = pd.read_csv(sio, sep=',')
    
    truth = pd.concat([ntd, nntd])
    truth = truth.set_index('filename')
    return truth
    

# Load clinicial diagnoses