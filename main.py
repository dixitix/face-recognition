%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

# use seaborn plotting defaults
import seaborn as sns; sns.set()

import glob
from google.colab import drive
drive.mount('/gdrive', force_remount=True)

from tqdm import tqdm

dirname = "/gdrive/My Drive/train"

count = 0

Xtrain, ytrain = [], []

for fname in tqdm(os.listdir(dirname)):
    img = cv2.imread(os.path.join(dirname, fname), cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    face = face_cascade.detectMultiScale(img)
    for (a, b, w, h) in face:
        cv2.rectangle(img, (a, b), (a+w, b+h), (0, 0, 255), 2)
        face = img[b:b + h, a:a + w]

    try:
        h, w = face.shape
        size = min(h, w)
        h0 = int((h - size) / 2)
        w0 = int((w - size) / 2)
    
        img = face[h0: h0 + size, w0: w0 + size]
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
    
        Xtrain.append(img)
        ytrain.append(int(fname.split('label')[1].split('.jpg')[0]))
    except:
        count += 1

len(Xtrain), len(ytrain)
count

import random
import albumentations as A

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

def get_aug_noise(image):
  transform = A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ])
  return transform(image=image)['image']
  
def get_aug_contrast(image):
  transform = A.OneOf([
            A.RandomBrightnessContrast(),  
            A.IAASharpen(),
            A.IAAEmboss(),     
        ])
  return transform(image=image)['image']
  
  
def get_aug_blur(image):
   transform = A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ])
   return transform(image=image)['image']
   
def get_aug_angle_and_shift(image):
    angle = np.arange(-20,20,1)
    angle0 = random.choice(angle)

    shift = 0.01*np.arange(-20,20,1)
    shift0 = random.choice(shift)

    transform = A.ShiftScaleRotate(shift_limit=shift0,rotate_limit=angle0,scale_limit=0,p=0.5)

    return transform(image=image)['image']
    
 def get_aug_flip(image):
  transform = A.HorizontalFlip(p = 0.5)
  return transform(image=image)['image']
  
XtrainAug = []
ytrainAug = []

for (a,b) in zip(Xtrain,ytrain):
    for i in range(3):
      XtrainAug.append(get_aug_noise(a))
      ytrainAug.append(b)
    for i in range(3):
      XtrainAug.append(get_aug_contrast(a))
      ytrainAug.append(b)
    for i in range(3):
      XtrainAug.append(get_aug_blur(a))
      ytrainAug.append(b)
    for i in range(3):
      XtrainAug.append(get_aug_angle_and_shift(a))
      ytrainAug.append(b)
    for i in range(3):
      XtrainAug.append(get_aug_flip(a))
      ytrainAug.append(b)
      
XtrainAug = np.asarray([el.ravel() for el in XtrainAug])
XtrainAug.shape

from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA #Principal Components Analysis
from sklearn.pipeline import make_pipeline

pca = KernelPCA(n_components=200, kernel='poly', random_state=42)

model_svm2 = SVC(kernel = 'rbf', C=12)

model_svm2.fit(XtrainAug, ytrainAug)

from sklearn.metrics import accuracy_score, f1_score

pred_svm2 = model_svm2.predict(XtrainAug)

accuracy_score(ytrainAug, pred_svm2)

def sort_by_index(fname):
    return int(fname.split('.')[0].split('img')[1])
    
from tqdm import tqdm

dirname = "/gdrive/My Drive/test"

Xtest = []
Names = []

for fname in tqdm(sorted(os.listdir(dirname), key=sort_by_index)):
    img = cv2.imread(os.path.join(dirname, fname), cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    face = face_cascade.detectMultiScale(img)

    Names.append(fname)

    for (a, b, w, h) in face:
        cv2.rectangle(img, (a, b), (a+w, b+h), (0, 0, 255), 2)
        face = img[b:b + h, a:a + w]

    try:
        h, w = face.shape
        size = min(h, w)
        h0 = int((h - size) / 2)
        w0 = int((w - size) / 2)
    
        img = face[h0: h0 + size, w0: w0 + size]
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
    
        Xtest.append(img)
    except:
        h, w = img.shape
        size = min(h, w)
        h0 = int((h - size) / 2)
        w0 = int((w - size) / 2)
            
        img = img[h0: h0 + size, w0: w0 + size]
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
            
        Xtest.append(img)

len(Xtest)

Xtest = np.asarray([el.ravel() for el in Xtest])
pred_lr = model_svm2.predict(Xtest)

import pandas as pd

pred_df = pd.DataFrame(list(zip(Names, pred_lr)), columns = ['img', 'label'])
pred_df.head()

pred_df.to_csv("huur5.csv", index=False)

from sklearn.model_selection import cross_val_score

bad_scores = cross_val_score(model_svm2, XtrainAug, ytrainAug, cv=15)

bad_scores
