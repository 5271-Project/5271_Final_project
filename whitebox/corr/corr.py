#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import resnet
import h5py
import cv2
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

from data import load_data
from resnet import ResNet

img_dir = os.path.join(os.getcwd(), 'test_imgs')

model_save_dir = os.path.join(os.getcwd(), 'saved_models')
res_model_save_name = 'saved_res_corr_model.hdf5'
res_model_save_path = os.path.join(model_save_dir, res_model_save_name)

cor_level = 1
extract_length = 32 * 32 * 3 * 1

def extract_params(extract_length, total_params):
  rest_length = extract_length
  params = K.constant([])
  for layer_weights in total_params:
    layer_weights_flat = K.flatten(layer_weights)

    if layer_weights_flat.shape[0] < rest_length:
      params = K.concatenate((params, layer_weights_flat))
    else:
      if rest_length > 0:
        params = K.concatenate((params, layer_weights_flat[:rest_length]))
      break
    rest_length = rest_length - layer_weights_flat.shape[0]

  return params

class Callbacks(keras.callbacks.Callback):
  def __init__(self, _total_weights):
    self.total_weights = _total_weights 
  def on_batch_begin(self, batch, logs):
    K.set_value(self.total_weights, 
            extract_params(extract_length, self.model.get_weights()))
  # def on_batch_end(self, batch, logs):
  #   print(logs['loss'])

save_res_model = ModelCheckpoint(
        res_model_save_path,
        monitor='val_acc',
        verbose=1,
        mode='max')


class CorrRes:
  def __init__(
          self, 
          input_shape, 
          classes, 
          epochs, 
          batch_size,
          extract_length = 32 * 32 * 3 * 1,
          optimizer = 'adam',
          dataset = 'cifar'):

    self.input_shape = input_shape
    self.classes = classes
    self.model = self.get_model()
    self.epochs = epochs
    self.batch_size = batch_size
    self.extract_length = extract_length
    self.optimizer = optimizer
    self.dataset = dataset

    self.loss_value = None
    self.acc_value = None

    self.train_data, self.test_data = load_data(
            name = self.dataset, classes = self.classes)

    self.extracted_data = self.train_data[0].flatten()[:self.extract_length]
    #self.total_weights = self.model.get_weights()
    self.total_weights = K.variable(
            extract_params(self.extract_length, self.model.get_weights()))

    self.model = self.compile_model()

  def get_model(self):
    self.model = ResNet(
      input_shape = self.input_shape, 
      classes = self.classes)
    return self.model

  def compile_model(self):
    self.model.compile(optimizer = self.optimizer,
                  loss = self.loss(),
                  metrics = ['accuracy'])
    return self.model

  def loss(self):
    def loss_function(y_true, y_pred):

      params = K.cast(self.total_weights, dtype='float32')
      target_data = K.cast(self.extracted_data, dtype='float32')
      params_mean = K.mean(params)
      target_mean = K.mean(target_data)
      params_d = params - params_mean
      target_d = target_data - target_mean

      num = K.sum((params_d) * (target_d))
      den = K.sqrt(K.sum(K.square(params_d)) * K.sum(K.square(target_d)))
      co = num / den
      loss_co = 1 - abs(co)

      # print("")
      # K.print_tensor(loss_co, message="params=")
      # K.print_tensor(loss_co, message="data=")

      loss = keras.losses.categorical_crossentropy(y_true, y_pred)
      # K.print_tensor(loss, message="loss=")
      # return K.mean(K.abs(y_true - y_pred))
      # Used loss - abs(co) before
      # return loss + loss_co * cor_level

      self.loss_value = loss + loss_co * cor_level
      return self.loss_value
    
    return loss_function
 
  def train(self):
    #K.print_tensor(extracted_data, message="fff=")
    # model.add_loss()

    self.model.fit(
            self.train_data[0], 
            self.train_data[1], 
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            callbacks=[Callbacks(self.total_weights), save_res_model])

  def load(self, model_path):
    self.model.load_weights(model_path)
    self.total_weights = K.variable(
            extract_params(self.extract_length, self.model.get_weights()))
    pass

  def evaluate(self):
    result = self.model.evaluate(self.test_data[0], self.test_data[1])
    self.loss_value = result[0]
    self.accuracy = result[1]

  def attack_evaluate(self):
    img_name = 'test.png'
    img_path = os.path.join(img_dir, img_name)

    data = self.extracted_data.reshape(32, 32, 3)

    cv2.imwrite(img_path, data)

    pass

if __name__ == '__main__':
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)

  corr_res = CorrRes((32, 32, 3), 10, 1, 128)
  # corr_res.train()
  corr_res.load(res_model_save_path) 
  corr_res.attack_evaluate()
