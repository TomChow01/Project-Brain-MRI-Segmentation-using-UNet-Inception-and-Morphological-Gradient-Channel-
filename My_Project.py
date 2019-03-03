# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:20:10 2019

@author: TAMAL
"""

import os
import numpy as np
#import os
import nibabel as nib
import cv2
import h5py
import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

Dataset='C:/Users/hp/Desktop/ML Projects/Adversirial ML/gm_wm/MRBraiN13/MRBrainS13DataNii/TrainingData'
T1 = []
T1_1mm = []
T1_IR=[]
T2_FLAIR=[]
OT=[]

p=os.listdir(Dataset)
for p1 in p:
    T1.append(glob.glob(Dataset+'/'+ p1+ '/*T1.nii*') )
    T1_1mm.append(glob.glob(Dataset+'/'+p1+'/*T1_1mm.nii*'))
    T1_IR.append(glob.glob(Dataset+'/'+p1+'/*T1_IR.nii*'))
    T2_FLAIR.append(glob.glob(Dataset+'/'+p1+'/*T2_FLAIR.nii*'))
    OT.append(glob.glob(Dataset+'/'+p1+'*/LabelsForTraining.nii*'))
data_T1=[]
data_T1_1mm=[]
data_T1_IR=[]
data_T2_FLAIR=[]
data_OT=[]

for i in range(len(T1) ):
    X = nib.load(T1[i][0])
    img=X.get_fdata()
    img=np.asarray(img,'float32')
    img=img.T/np.max(img)
#    img=1-img
#    F=np.argwhere(img==1)
#    img[np.squeeze(F[:,0]),np.squeeze(F[:,1]),np.squeeze(F[:,2])]=0
    img=(img-np.mean(img))/(np.std(img)+0.00001)
#    img=(img-np.min(img)+0.00001)/(np.max(img)-np.min(img)+0.00001)
    data_T1.append(img)
    
    X = nib.load(T1_1mm[i][0])
    img=X.get_fdata()
    img=np.asarray(img,'float32')
    img=img.T/np.max(img)
#    img = img(np.sum(img[i])>0 for i in range (img.shape[0]))

    img=(img-np.mean(img))/(np.std(img)+0.00001)
#    img=(img-np.min(img)+0.00001)/(np.max(img)-np.min(img)+0.00001)
#    img=1-img
#    F=np.argwhere(img==1)
#    img[np.squeeze(F[:,0]),np.squeeze(F[:,1]),np.squeeze(F[:,2])]=0
    data_T1_1mm.append(img)
    
    X = nib.load(T1_IR[i][0])
    img=X.get_fdata()
    img=np.asarray(img,'float32')
    img=img.T/np.max(img)
#    img = img(np.sum(img[i])>0 for i in range (img.shape[0]))
    
    img=(img-np.mean(img))/(np.std(img)+0.00001)
#    img=(img-np.min(img)+0.00001)/(np.max(img)-np.min(img)+0.00001)
#    img=1-img
#    F=np.argwhere(img==1)
#    img[np.squeeze(F[:,0]),np.squeeze(F[:,1]),np.squeeze(F[:,2])]=0
    data_T1_IR.append(img)
    
    X = nib.load(T2_FLAIR[i][0])
    img=X.get_fdata()
    img=np.asarray(img,'float32')
    img=img.T/np.max(img)
#    img = img(np.sum(img[i])>0 for i in range (img.shape[0]))

    img=(img-np.mean(img))/(np.std(img)+0.00001)
#    img=(img-np.min(img)+0.00001)/(np.max(img)-np.min(img)+0.00001)
#    img=1-img
#    F=np.argwhere(img==1)
#    img[np.squeeze(F[:,0]),np.squeeze(F[:,1]),np.squeeze(F[:,2])]=0
    data_T2_FLAIR.append(img)
    
    X = nib.load(OT[i][0])
    img=X.get_fdata()
    img=np.asarray(img,'float32')
    img=img.T
    data_OT.append(img)

def show(data,img_num,slide):
    plt.imshow(data[img_num][slide,:,:])
    plt.show()




#------------------------------------------------------Augmentation--------------------------------------------
def aug_img(data,num):
    args_cls_1 = dict(
                rotation_range=10,
                width_shift_range=0.0,
                height_shift_range=0.0,
                shear_range=0.0,
                zoom_range=0.0,
                horizontal_flip=True,
                fill_mode='nearest')
        
    args_cls_2 = dict(
                rotation_range=00,
                width_shift_range=0.0,
                height_shift_range=0.0,
                shear_range=0.0,
                zoom_range=0.0,
                vertical_flip=True,
                fill_mode='nearest')
    
    args_cls_3 = dict(
                rotation_range=10,
                width_shift_range=0.0,
                height_shift_range=0.0,
                shear_range=0.0,
                zoom_range=0.0,
                horizontal_flip=False,
                fill_mode='nearest')
    
    seed = 1
    
    img_datagen_cls_1 = ImageDataGenerator(**args_cls_1)
    img_datagen_cls_2 = ImageDataGenerator(**args_cls_2)
    img_datagen_cls_3 = ImageDataGenerator(**args_cls_3)
        
    
#    mask_datagen_cls_1 = ImageDataGenerator(**args_cls_1)
#    mask_datagen_cls_2 = ImageDataGenerator(**args_cls_2)
#    mask_datagen_cls_3 = ImageDataGenerator(**args_cls_3)
    
    img_datagen_cls_1.fit(data, augment=True, seed=seed)
    img_datagen_cls_2.fit(data, augment=True, seed=seed)
    img_datagen_cls_3.fit(data, augment=True, seed=seed)
    

    img_cls_1_augmented = img_datagen_cls_1.flow(np.asarray(data), batch_size = 1, seed=seed)
    img_cls_2_augmented = img_datagen_cls_2.flow(np.asarray(data), batch_size = 1, seed=seed)
    img_cls_3_augmented = img_datagen_cls_3.flow(np.asarray(data), batch_size = 1, seed=seed)

    
    data_aug_cls_1 = [next(img_cls_1_augmented)[0] for i in range(num)]
    data_aug_cls_2 = [next(img_cls_2_augmented)[0] for i in range(num)]
    data_aug_cls_3 = [next(img_cls_3_augmented)[0] for i in range(num)]
    
    data_aug = np.concatenate((data_aug_cls_1,data_aug_cls_2,data_aug_cls_3))
    
    return data_aug

data_T1_aug = aug_img(data_T1,10)
data_T1_1mm_aug = aug_img(data_T1_1mm,10)
data_T1_IR_aug = aug_img(data_T1_IR,10)
data_T2_FLAIR_aug = aug_img(data_T2_FLAIR,10)
data_OT_aug = aug_img(data_OT,10)





    


##--------------------------------------2part division-----------------------------------------------
hf=h5py.File('Test_SISS.h5', 'r')
print(list(hf.keys()))
X_Data=hf['X_Data_T'][:]
X_Data1=hf['X_Data_F'][:]
pT=hf['patch_T'][:]
pf=hf['patch_F'][:]


hf.close()
