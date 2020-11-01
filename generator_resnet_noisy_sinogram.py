#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:01:45 2020

@author: admin
"""



import tensorflow as tf
import pickle
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import matplotlib.pyplot as plt
data = pickle.load(open('sinogram2.pickle','rb'))     
imageset = data['imageset']
sinogram = data['sinogram']   
sinogram_noisy = data['sinogram_noisy']   



inx = np.random.permutation(sinogram_noisy.shape[0])
inx_ = int(0.9*len(inx))
inx_train = inx[:inx_]
inx_test = inx[inx_:]
OUTPUT_CHANNELS = 3

mode = 'clean'

## ------------ preprocessing data
if mode  is 'noisy':
    scale_value = np.max([np.max(sinogram_noisy), np.max(sinogram)])
    Xdata = sinogram_noisy.copy()
else:
    scale_value =  np.max(sinogram)
    Xdata = sinogram.copy()


Xdata = Xdata/scale_value*128
Xdata = Xdata[:,:,:,:,np.newaxis] + np.zeros((1,1,1,1,3),dtype='float32')

Ydata = sinogram[:,:1].copy()
Ydata = Ydata/scale_value*128
Ydata = Ydata[:,:,:,:,np.newaxis] + np.zeros((1,Xdata.shape[1],1,1,3),dtype='float32')

# data = {'Xdata':Xdata, 'Ydata':Ydata, 'inx_train':inx_train, 'inx_test':inx_test}
# pickle.dump(data,open('noisy_sinogram_processed_for_Resnet.pickle','wb'))



##########################################
# data = pickle.load(open('noisy_sinogram_processed_for_Resnet.pickle','rb'))     
# Xdata = data['Xdata'] 
# Ydata = data['Ydata']
# inx_train = data['inx_train']
# inx_test = data['inx_test']


# 












################### gradual training with autoencoder MAE #############3
generator = generator_resnet(OUTPUT_CHANNELS)
opt = tf.keras.optimizers.Adam()



def generator_loss(y_true, y_pred, input_image, Lambda):
    
    diff1 = tf.abs(y_true-y_pred)
    diff2 = tf.square(y_true-y_pred)
    idx1 = tf.where(input_image>0)
    idx2 = tf.where(input_image==0)
    loss_sampledpx = tf.reduce_mean(tf.gather_nd(diff1,idx1))
    loss_unsampledpx = tf.reduce_mean(tf.gather_nd(diff2,idx2))
    total_loss = loss_sampledpx*Lambda +  (1-Lambda)*loss_unsampledpx
            
        
    return total_loss, loss_sampledpx, loss_unsampledpx



@tf.function
def train_step(train_image, target,Lambda):
  with tf.GradientTape() as gen_tape:
    y_pred = generator(train_image, training=True)
    total_loss, loss_sampledpx, loss_unsampledpx = generator_loss(target, y_pred, train_image,Lambda)    

  gradients = gen_tape.gradient(total_loss, generator.trainable_variables) 
  opt.apply_gradients(zip(gradients, generator.trainable_variables))
  return total_loss, loss_sampledpx, loss_unsampledpx

@tf.function
def validate(input_image, target, Lambda):  
    y_pred = generator(input_image, training=False)
    total_loss, loss_sampledpx, loss_unsampledpx = generator_loss(target, y_pred, input_image,Lambda) 
    return total_loss, loss_sampledpx, loss_unsampledpx

def fit(train_ds, epochs, Lambda,test_ds):    
    hist=dict()
    hist_tr={'tloss':[],'sampledpx_loss':[],'unsampledpx_loss':[]}
    hist_val={'tloss':[],'sampledpx_loss':[],'unsampledpx_loss':[]}
    
    x_test, y_test = test_ds    
    for epoch in range(epochs):
        # Train
        tl =0;
        loss_s =0;
        loss_uns =0;
        for n, (input_image, target) in train_ds.enumerate():
            total_loss, loss_sampledpx, loss_unsampledpx = train_step(input_image, target,Lambda)
            tl += total_loss
            loss_s += loss_sampledpx
            loss_uns += loss_unsampledpx
            print('.', end='')
        tl = tl.numpy()/float(n+1)
        loss_s = loss_s.numpy()/float(n+1)
        loss_uns = loss_uns.numpy()/float(n+1)
        hist_tr['tloss'].append(tl)
        hist_tr['sampledpx_loss'].append(loss_s)
        hist_tr['unsampledpx_loss'].append(loss_uns)
        print()
        print('epoch:{}/{}, tloss:{:.3f}, sampled:{:.3f}, unsampled:{:.3f}'.\
              format(epoch, epochs, total_loss, loss_sampledpx, loss_unsampledpx))
            
            
        vtotal_loss, vloss_sampledpx, vloss_unsampledpx = validate(x_test, y_test, Lambda)       
        print('valtloss:{:.3f}, val_sampled:{:.3f}, val_unsampled:{:.3f}'.\
              format(vtotal_loss, vloss_sampledpx, vloss_unsampledpx))
        hist_val['tloss'].append(vtotal_loss.numpy())
        hist_val['sampledpx_loss'].append(vloss_sampledpx.numpy())
        hist_val['unsampledpx_loss'].append(vloss_unsampledpx.numpy())
    hist['train'] = hist_tr
    hist['val'] = hist_val
    return hist
  
  #########################################
# x_train = Xdata[inx_train]
# y_train = Ydata[inx_train]
# x_train = np.reshape(x_train,(-1,224,224,3))
# y_train = np.reshape(y_train,(-1,224,224,3))
    
    
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(300)
# train_dataset = train_dataset.batch(108)


x_test = Xdata[inx_test,3:]
y_test = Ydata[inx_test,3:]
x_test = np.reshape(x_test,(-1,224,224,3))
y_test = np.reshape(y_test,(-1,224,224,3))
test_ds = (x_test, y_test)




hist_tr={'tloss':[],'sampledpx_loss':[],'unsampledpx_loss':[]}
hist_val={'tloss':[],'sampledpx_loss':[],'unsampledpx_loss':[]}
lr_steps =[1e-2, 1e-2]
historys =[]
# for i,j in enumerate(range(0,4)):
for i in range(len(lr_steps)):    
    j=3
    x_train = Xdata[inx_train,j:j+2]
    y_train = Ydata[inx_train,j:j+2]
    x_train = np.reshape(x_train,(-1,224,224,3))
    y_train = np.reshape(y_train,(-1,224,224,3))
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(300)
    train_dataset = train_dataset.batch(108)

    
    opt.learning_rate = lr_steps[i]
    if i>0:
        epochs =10
        Lambda = 0.9
    else:
        epochs =10
        Lambda = 0.1
    hist = fit(train_dataset, epochs, Lambda,test_ds)
    hist_tr['tloss'].extend(hist['train']['tloss'])
    hist_tr['sampledpx_loss'].extend(hist['train']['sampledpx_loss'])
    hist_tr['unsampledpx_loss'].extend(hist['train']['unsampledpx_loss'])
    hist_val['tloss'].extend(hist['val']['tloss'])
    hist_val['sampledpx_loss'].extend(hist['val']['sampledpx_loss'])
    hist_val['unsampledpx_loss'].extend(hist['val']['unsampledpx_loss'])

  
###################################################333
# model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError())
# lr_steps =[1e-3, 1e-3, 1e-4, 1e-4]
# historys =[]
# for i,j in enumerate(range(0,4)):
#     x_train = Xdata[inx_train,j:j+2]
#     y_train = Ydata[inx_train,j:j+2]
#     x_train = np.reshape(x_train,(-1,224,224,3))
#     y_train = np.reshape(y_train,(-1,224,224,3))
    
#     model.optimizer.learning_rate = lr_steps[i]
#     history = model.fit(x_train,y_train, batch_size=108, epochs=10, shuffle=True,
#                     validation_data=(x_test, y_test))
#     historys.append(history)



################## plot
nsub_sample = Xdata.shape[1]
sub_sample_idx =1

y_test_sub = y_test[slice(sub_sample_idx,-1,nsub_sample)]
x_test_sub = x_test[slice(sub_sample_idx,-1,nsub_sample)]
pred_test = generator(x_test_sub)
pred_test = pred_test.numpy()

gt = y_test_sub[:,:,:,0].flatten()
xsam = x_test_sub[:,:,:,0].flatten()
pred = pred_test[:,:,:,0].flatten()
idx = xsam!=0

err1 = np.mean(abs(gt[idx]-xsam[idx]))
err2 = np.mean(abs(gt[idx]-pred[idx]))

err1 = np.mean(abs(gt[:]-xsam[:]))
err2 = np.mean(abs(gt[:]-pred[:]))

idx=1 #2
full_sinogram = y_test_sub[idx,:,:,0]
subsample_sinogram = x_test_sub[idx,:,:,0]
pred_sinogram = pred_test[idx,:,:,0]
nang = pred_sinogram.shape[1]
theta0 = np.linspace(0., 180., nang, endpoint=False)


from skimage.transform import iradon
plt.figure(figsize=(15,15))
ax= plt.subplot(131)
recon = iradon(full_sinogram, theta=theta0, circle=True)
ax.matshow(recon)
ax.axis('off')
plt.title('recon_full_sinogram')

ax=plt.subplot(132)
idx = np.where(np.sum(subsample_sinogram,axis=0)>0)[0]
recon1 = iradon(subsample_sinogram[:,idx], theta=theta0[idx], circle=True)
ax.matshow(recon1)
ax.axis('off')
plt.title('subsample_sinogram')

ax= plt.subplot(133)
recon2 = iradon(pred_sinogram, theta=theta0, circle=True)
ax.matshow(recon2)
ax.axis('off')
plt.title('subsample_generator_sinogram')