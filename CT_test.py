#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 00:41:11 2020

@author: admin
"""


from skimage.transform import iradon_sart


import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from skimage.transform import iradon
from skimage.transform import resize

image = shepp_logan_phantom()
ss =[]
sIs=[]
recons =[]
reconIs =[]
for numViews in [360, 180, 90, 45, 20]:
    theta = np.linspace(0., 360., int(numViews), endpoint=False)
    s= radon(image, theta=theta, circle=True)
    recon = iradon(s, theta=theta, circle=True)
    recons.append(recon)
    ss.append(s)
    sI = resize(s,(s.shape[0],360),anti_aliasing=True)
    sIs.append(sI)
    
    theta = np.linspace(0., 360., 360, endpoint=False)
    reconI = iradon(sI, theta=theta, circle=True)
    reconIs.append(reconI)


nang = 720
hs_angle = 180/nang

theta0 = np.linspace(0., 180., nang, endpoint=False)
S0 = radon(image, theta=theta0, circle=True)
reconI = iradon(S0, theta=theta0, circle=True)
# np.sqrt(sum(sum((image-reconI)**2)))


##########################################################
from skimage import transform
import math

ang = math.pi/4
sv = 1
yb =100
xb = 0

def image_transform(image, sv, ang, xb, yb):
    """
    transform image scale, rotation, translation
    Parameters
    ----------
    image : 2d-array image        
    sv : scale value        
    ang : angle in radian        
    xb : bias in x        
    yb : bias in y        

    Returns
    -------
    image1 : 2d-array image         

    """
    assert image.shape[0]==image.shape[1], "image should be a square"
    center = image.shape[0]//2
    x0  = center * (-math.cos(ang) +math.sin(ang))/sv +center
    y0  = center * (-math.cos(ang) -math.sin(ang))/sv +center
    tform = transform.SimilarityTransform(scale=1/sv, rotation= ang,
                                          translation=(x0+xb, y0+yb))
    
    image1 = transform.warp(image, tform)
    
    
    #shape_min = image.shape[0]
    radius = image.shape[0] // 2
    img_shape = np.array(image.shape)
    coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]])
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_disk = dist > radius ** 2
    image1[outside_disk]=0    
    return image1


# nang = 360
# full_theta = np.linspace(0., 180., nang, endpoint=False)
# fullsinogram = radon(image, theta=full_theta, circle=True)

def gen_sparse_sinogram(fullsinogram, full_theta, sampling_interval):
    #sampling_interval = 4
    #slice_idx = slice(0,720,sampling_interval)
    slice_idx = list(range(0,len(full_theta),sampling_interval))
    
    sinogram = np.zeros(fullsinogram.shape)
    sinogram[:, slice_idx]= fullsinogram[:, slice_idx]
    return sinogram, slice_idx



####################### use Unet
import tensorflow as tf
import pickle
from tensorflow_examples.models.pix2pix import pix2pix

image = resize(shepp_logan_phantom(),(224,224))


scales = [ 0.5, 0.7, 0.9, 1]
angs = np.linspace(0,180,30)/180*math.pi
nrand = 10
imageset = np.empty((len(scales)*len(angs)*nrand,)+image.shape)
k=0
for sv in scales:
    for ang in angs:
        for i in range(nrand):
            xb = np.random.randint(-100,100)
            yb = np.random.randint(-100,100)
            img1 = image_transform(image, sv, ang, xb, yb)
            imageset[k,:,:]=img1
            k+=1


# data = pickle.load(open('sinogram.pickle','rb'))     
# imageset = data['imageset']   
# imageset= imageset.astype('float32')

nimg = imageset.shape[0]
samp_interval =[1,2,4,6,8]
nsamp = len(samp_interval)
sinogram = np.empty([nimg, nsamp, imageset.shape[1], imageset.shape[2]],dtype='float32')


nang = 224
full_theta = np.linspace(0., 180., nang, endpoint=False)
for iimg in range(nimg):   
    image = imageset[iimg]
    fullsinogram = radon(image, theta=full_theta, circle=True)      
    for isamp, interval in enumerate(samp_interval):
        partial, slice_idx = gen_sparse_sinogram(fullsinogram, full_theta, interval)
        sinogram[iimg,isamp,:,:] = partial

data = {'imageset':imageset, 'sinogram':sinogram, 'full_theta':full_theta, 'samp_interval':samp_interval}
pickle.dump(data,open('sinogram.pickle','wb'))

####################################################3

import tensorflow as tf
import pickle
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import matplotlib.pyplot as plt
data = pickle.load(open('sinogram.pickle','rb'))     
sinogram = data['sinogram']   




inx = np.random.permutation(sinogram_.shape[0])
inx_ = int(0.9*len(inx))
inx_train = inx[:inx_]
inx_test = inx[inx_:]
OUTPUT_CHANNELS = 3

##-------------- mobilenet base
# sinogram_= sinogram.copy()
# sinogram_ = sinogram_/np.max(sinogram_.flatten())
# sinogram_ = sinogram_[:,:,:,:,np.newaxis] + np.zeros((1,1,1,1,3),dtype='float32')
# model = unet_model(OUTPUT_CHANNELS)
# model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

## ------------ resnet base
sinogram_= sinogram.copy()
sinogram_ = sinogram_/np.max(sinogram_.flatten())*125
sinogram_ = sinogram_[:,:,:,:,np.newaxis] + np.zeros((1,1,1,1,3),dtype='float32')
# data = {'sinogram_':sinogram_,  'inx_train':inx_train, 'inx_test':inx_test}
# pickle.dump(data,open('sinogram_processed_for_Resnet.pickle','wb'))

data = pickle.load(open('sinogram.pickle','rb'))     
sinogram_ = data['sinogram_'] 
inx_train = data['inx_train']
inx_test = data['inx_test']


# 





x_test = sinogram_[inx_test,1:5]
y_test = sinogram_[inx_test,:1]+np.zeros((1,4,1,1,1),dtype='float32')
x_test = np.reshape(x_test,(-1,224,224,3))
y_test = np.reshape(y_test,(-1,224,224,3))

################################ autoencoder MSE ###################
model = unet_resnet(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
x_train = sinogram_[inx_train,1:5]
y_train = sinogram_[inx_train,:1]+np.zeros((1,3,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=50, shuffle=True,
                validation_data=(x_test, y_test))
model.save('./testMSE.model')


################################ autoencoder MSE gradual learning ###################
model = unet_resnet(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
x_train = sinogram_[inx_train,:2]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=20, shuffle=True,
                validation_data=(x_test, y_test))

x_train = sinogram_[inx_train,1:3]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=20, shuffle=True,
                validation_data=(x_test, y_test))

x_train = sinogram_[inx_train,3:5]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=20, shuffle=True,
                validation_data=(x_test, y_test))


model.save('./testMSE_gradual.model')


################################ autoencoder MAE ###################
model = unet_resnet(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())
x_train = sinogram_[inx_train,1:5]
y_train = sinogram_[inx_train,:1]+np.zeros((1,3,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=50, shuffle=True,
                validation_data=(x_test, y_test))
model.save('./testMAE.model')


################### gradual training with autoencoder MAE #############3

model = unet_resnet(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())
x_train = sinogram_[inx_train,:2]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=20, shuffle=True,
                validation_data=(x_test, y_test))

x_train = sinogram_[inx_train,1:3]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=20, shuffle=True,
                validation_data=(x_test, y_test))

x_train = sinogram_[inx_train,3:5]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=10, shuffle=True,
                validation_data=(x_test, y_test))

model.save('./testMAE_gradual.model')
#######################   MAE with only sparse sinogram  #####################3

# This does not work and do not reduce validation error
model = unet_resnet(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())
x_train = sinogram_[inx_train,3:5]
y_train = sinogram_[inx_train,:1]+np.zeros((1,2,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))
history = model.fit(x_train,y_train, batch_size=108, epochs=50, shuffle=True,
                validation_data=(x_test, y_test))


model.save('./testMAE_onlysparse.model')



# model = tf.keras.models.load_model('./test.model')

##########################

x_test = sinogram_[inx_test]
x_test1 = x_test[:,0].squeeze()



x_test2 = x_test[:,1].squeeze()
x_test3 = x_test[:,2].squeeze()
x_test4 = x_test[:,3].squeeze()
x_test5 = x_test[:,4].squeeze()
pred_test = model(x_test5)
pred_test = pred_test.numpy()

idx=30
full_sinogram = x_test1[idx,:,:,0]
sparse_sinogram = x_test5[idx,:,:,0]
pred_sinogram = pred_test[idx,:,:,0]
nang = pred_sinogram.shape[1]
theta0 = np.linspace(0., 180., nang, endpoint=False)

plt.figure()
ax=plt.subplot(2,2,1)
idx = np.where(np.sum(sparse_sinogram,axis=0)>0)[0]
recon = iradon(sparse_sinogram[:,idx], theta=theta0[idx], circle=True)
ax.matshow(recon)
ax= plt.subplot(222)
recon = iradon(pred_sinogram, theta=theta0, circle=True)
ax.matshow(recon)
ax= plt.subplot(223)
recon = iradon(full_sinogram, theta=theta0, circle=True)
ax.matshow(recon)


plt.figure()
ax=plt.subplot(2,2,1)
ax.matshow(sparse_sinogram)
ax= plt.subplot(222)
ax.matshow(pred_sinogram)
ax= plt.subplot(223)
ax.matshow(full_sinogram)


###################33

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = train_dataset.batch(2)

for example_input, example_target in train_dataset.take(1):
    plt.matshow(example_input[:,:,0])







############################### calculate system matrix from image to sinogram
from skimage.transform import warp_coords,resize, ProjectiveTransform, warp
from scipy.ndimage import map_coordinates
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
import numpy as np
# x = np.linspace(0,19,20)
# y = np.linspace(0,19,20)

# X,Y= np.meshgrid(x,y,indexing='xy')

# xy = np.array([Y.flatten(),X.flatten()]).T


def rotate(ang, center):    
    """
    inverse rotation centered at the image center

    Parameters
    ----------
    ang : angle in degree
    center : image center

    Returns
    -------
    R : affine matrix for rotation

    """
    angle=np.deg2rad(ang)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                  [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                  [0, 0, 1]])
    return R


def get_rot_coordinates(R, shape):
    """
    image rotation coordinates calculation with affine matrix R
    
    Parameters
    ----------
    R : affine matrix 3x3        
    shape : image_shape
        

    Returns
    -------
    coords_index : coordinate_index with order = 'C'       
    idx_valid : valid index

    """
    inverse_map = ProjectiveTransform(matrix=R)
    coords = warp_coords(inverse_map, shape)
    xs=coords[0].flatten().astype('int')
    ys=coords[1].flatten().astype('int')
    
    a=np.all(np.vstack((xs>=0, xs<shape[1],ys>=0, ys<shape[0])), axis=0)
    idx_valid = np.where(a)[0]
    
    coords_index = np.nan*np.ones((np.prod(shape),))
    coords_index[idx_valid]=1
    

    coords_index[idx_valid] = np.ravel_multi_index((xs[idx_valid],ys[idx_valid]), shape)
    coords_index  = np.reshape(coords_index, shape)
    return coords_index, idx_valid
    

theta0 = np.linspace(0., 180., 224, endpoint=False)
A = np.empty((224*224,224*224),dtype='uint8')
image = resize(shepp_logan_phantom(),(224,224))
image_size = image.shape[0]   # I assume image is square
# image_flatten = image.flatten()
for i, angle in enumerate(theta0):
    
    center = image_size//2
    R = rotate(angle, center)
    # rotated = warp(image, R, clip=False)
    # radon_image[:, i] = rotated.sum(0)
    
    
    coords_index, idx_valid = get_rot_coordinates(R, image.shape)
    
    # proj = np.zeros((coords_index.shape[1],1))
    for ic in range(image_size):
        idx_= np.where(~np.isnan(coords_index[:,ic]))
        idx = coords_index[idx_,ic].astype('int')        
        # proj[ic] = image_flatten[idx].sum()    
        k = i*image_size+ic
        A[k,idx]=1
        
    # plt.plot(rotated.sum(0))
    # plt.plot(proj)
    
# sino = radon(image, theta=theta0)        
# sino1 = A.dot(image_flatten)
# sino1 = np.reshape(sino1,image.shape)

# plt.figure()
# ax=plt.subplot(2,1,1)
# ax.matshow(sino)
# ax= plt.subplot(212)
# ax.matshow(sino1.T)


import h5py
hf = h5py.File('A.h5', 'w')

hf.create_dataset('A', data=A)
hf.close()


# R = rotate(90,10)

# image = np.random.randint(0,2,(5,5))
# image = resize(image,(21,21))


# inverse_map = ProjectiveTransform(matrix=R)
# coords = warp_coords(inverse_map, image.shape)
# wimg =  map_coordinates(image, coords)
# plt.matshow(wimg)
# plt.matshow(image)


# # direct rotation with R
# rotated = warp(image, R, clip=False)
# plt.matshow(rotated)
# plt.plot( rotated.sum(0))


# xs=coords[0].flatten().astype('int')
# ys=coords[1].flatten().astype('int')

# out_coords = np.ravel_multi_index((xs,ys), image.shape)
# out_coords = np.reshape(out_coords, image.shape)

# rotated=image.flatten()[out_coords]

# A = np.empty((224*224,224*224),dtype='uint8')


ATA = A.T.dot(A)


