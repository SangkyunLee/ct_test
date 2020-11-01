#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:16:25 2020

@author: sang
"""
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


#############   unet model based on mobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False
    

up_stack = [
    pix2pix.upsample(512, 3, apply_dropout=True),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3, apply_dropout=True),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3, apply_dropout=True),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3, apply_dropout=True),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[224, 224, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



#####################


resnet_model = tf.keras.applications.ResNet50(input_shape=[224, 224, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    # 'conv1_relu',            # 112x112
    'conv2_block3_out',      # 56x56
    'conv3_block4_out',      # 28x28
    'conv4_block6_out',      # 14x14
    'conv5_block3_out',      # 7x7
]
layers = [resnet_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=resnet_model.input, outputs=layers)

down_stack.trainable = False
    

up_stack = [
    pix2pix.upsample(1024, 3, apply_dropout=True),  # 7x7 -> 14x14
    pix2pix.upsample(512, 3, apply_dropout=True),  # 14x14 -> 28x28
    pix2pix.upsample(256, 3, apply_dropout=True),  # 28x28 -> 56x56
    # pix2pix.upsample(128, 3, apply_dropout=True),  # 56x56 -> 112x112
]

def unet_resnet(output_channels):
  inputs = tf.keras.layers.Input(shape=[224, 224, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  conv2dT = tf.keras.layers.Conv2DTranspose(
       256, 3, strides=2,
       padding='same')  #56x56 -> 112x112
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = conv2dT(x)
  x = last(x)  

  return tf.keras.Model(inputs=inputs, outputs=x)

###################################################33
def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


resnet_model = tf.keras.applications.ResNet50(input_shape=[224, 224, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    # 'conv1_relu',            # 112x112
    'conv2_block3_out',      # 56x56
    'conv3_block4_out',      # 28x28
    'conv4_block6_out',      # 14x14
    'conv5_block3_out',      # 7x7
]
layers = [resnet_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=resnet_model.input, outputs=layers)
down_stack.trainable = False
    

up_stack = [
    upsample(1024, 3, apply_dropout=True),  # 7x7 -> 14x14
    upsample(512, 3, apply_dropout=True),  # 14x14 -> 28x28
    upsample(256, 3, apply_dropout=True),  # 28x28 -> 56x56
    # upsample(128, 3, apply_dropout=True),  # 56x56 -> 112x112
]

def generator_resnet(output_channels):
  inputs = tf.keras.layers.Input(shape=[224, 224, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])
    
  x = upsample(128, 3, apply_dropout=True)(x)
  # This is the last layer of the model  
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #112x112 -> 128x128
  
  x = last(x)  

  return tf.keras.Model(inputs=inputs, outputs=x)






###############################################################
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[224, 224, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[224, 224, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 224, 224, channels*2)

  down1 = pix2pix.downsample(64, 4, apply_norm=False)(x) # (bs, 112, 112, 64)
  down2 = pix2pix.downsample(128, 4)(down1) # (bs, 56, 56, 128)
  down3 = pix2pix.downsample(256, 4)(down2) # (bs, 28, 28, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 30, 30, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 27, 27, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 29, 29, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 26, 26, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)





loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA=100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


#
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

#######################################
generator = unet_resnet(3)
discriminator = Discriminator()


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
   
    
import matplotlib.pyplot as plt   
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    ax= plt.subplot(1, 3, i+1)
    #plt.title(title[i])
    ax.matshow(display_list[i][:,:,0])
    # getting the pixel values between [0, 1] to plot it.
    #plt.imshow(display_list[i] * 0.5 + 0.5)
    #plt.axis('off')
  plt.show()
  
import time
from IPython import display

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)    
  
  
###################################3
x_test = sinogram_[inx_test,1:5]
y_test = sinogram_[inx_test,:1]+np.zeros((1,4,1,1,1),dtype='float32')
x_test = np.reshape(x_test,(-1,224,224,3))
y_test = np.reshape(y_test,(-1,224,224,3))


x_train = sinogram_[inx_train,:5]
y_train = sinogram_[inx_train,:1]+np.zeros((1,5,1,1,1),dtype='float32')
x_train = np.reshape(x_train,(-1,224,224,3))
y_train = np.reshape(y_train,(-1,224,224,3))


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(300)
train_dataset = train_dataset.batch(108)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(300)
test_dataset = test_dataset.batch(108)

EPOCHS =150  
fit(train_dataset, EPOCHS, test_dataset)


