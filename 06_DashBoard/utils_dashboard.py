

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, constraints, regularizers, models, optimizers
from tensorflow.keras.utils import Sequence
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt
import glob
import cv2

import time
from tqdm import tqdm

import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation, color
import pydicom
import imageio
from utils import *
from attention_layers import *


def loadSlices(patient_files):
    slices = [pydicom.read_file(s) for s in patient_files]
    slices.sort(key = lambda x: float(x.InstanceNumber))
        
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        try:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        except:
            slice_thickness = slices[0].SliceThickness
            slices[0].RescaleIntercept = 1024
    
    if slice_thickness == 0:
        slice_thickness = 1
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def getPixelsHu(patient_scans):
    patient_images = []
    for s in patient_scans:
        if s.Columns != s.Rows:
            crop_size = 512
            s_crop_img = imCropCenter(s.pixel_array, crop_size, crop_size)
            patient_images.append(s_crop_img)
        else:
            patient_images.append(s.pixel_array)

    patient_images = np.asarray(patient_images).astype(np.int16)

    # The intercept is usually -1024, so air is approximately 0
    patient_images[patient_images == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = patient_scans[0].RescaleIntercept
    slope = patient_scans[0].RescaleSlope
        
    if slope != 1:
        patient_images = slope * patient_images.astype(np.float64)
        patient_images = patient_images.astype(np.int16)

    patient_images += np.int16(intercept)
    patient_images = np.clip(patient_images, -2048, 3284)
    return np.array(patient_images, dtype=np.int16)


def plotHistogramPixelesHu(patient_images):
    plt.hist(patient_images.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel
    plt.show()
    

def plotSampleStack(stack, rows=6, cols=6, start_with=10, show_every=3, figsize=(12,12)):
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    return fig

 
## Interactive

def buildWindowGif(path, min_bound=-1_000, max_bound=400, name='scan_gif.gif'):
    images = np.load(path[0])
    images = windowImageNorm(images, min_bound=min_bound, max_bound=max_bound)
    imageio.mimsave("./tmp/" + name, images, duration=0.002)
    

# ****************************************************************************************** #
# Models

### Generator

class SequenceToSequenceDataGenerator(Sequence):
    
    def __init__(self, training, df, dict_ini_features, dict_patients_masks_paths,
                 batch_size=1, num_frames_batch=32, 
                 alpha=1.0, random_window=False, center_crop=True,
                 img_size_load=(500, 500, 3), 
                 img_size_crop=(440, 440, 3)):
        
        super(SequenceToSequenceDataGenerator, self).__init__()
        self.training = training
        self.df = df
        self.dict_ini_features = dict_ini_features
        self.batch_size = batch_size
        self.num_frames_batch = num_frames_batch
        self.alpha = alpha

        self.random_window = random_window
        self.center_crop = center_crop
        self.img_size_load = img_size_load
        self.img_size_crop = img_size_crop
        
        self.dict_patients_masks_paths = dict_patients_masks_paths
        
        self.ids = list(self.df['Patient'].unique())

        self.num_steps = int(np.ceil(len(self.ids) / self.batch_size))
        self.on_epoch_end()
      
    # Number of batches in the sequence
    
    def __len__(self):
        return self.num_steps
    
    
    # Gets the batch at position index, return patient images and dict ini features
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        patient_ids = [self.ids[k] for k in indexes]
        list_scan_imgs = [decodePatientImages(patient, 
                                              self.dict_patients_masks_paths,
                                              image_size=(self.img_size_load[0], self.img_size_load[1]), 
                                              numpy=True) 
                          for patient in patient_ids]
        patient_imgs = self.groupImages(list_scan_imgs)
        patient_imgs = self.loadImagesAugmented(patient_imgs)
        for patient_ in patient_ids:
            self.dict_ini_features[patient_]['Patient'] = patient_
        return patient_imgs, [self.dict_ini_features[patient_] for patient_ in patient_ids]
    
    # From n patient frames we will only keep self.alpha*n frames, cutting on top and bottom
    
    def filterSlices(self, array_imgs):
        num_patient_slices = array_imgs.shape[0]
        beta = int(self.alpha * num_patient_slices)
        if beta % 2 != 0:
            beta += 1
        if num_patient_slices > self.num_frames_batch:
            if beta > self.num_frames_batch and self.alpha < 1:
                remove = int((num_patient_slices - beta)/2)
                array_imgs = array_imgs[remove:, :, :, :]
                array_imgs = array_imgs[:-remove:, :, :]

        return array_imgs
    
    # Skip frames unniformally according to self.num_frames_batch value
    
    def frameSkipImages(self, patient_imgs):
        num_patient_slices = patient_imgs.shape[0]
        frame_skip = num_patient_slices // self.num_frames_batch
        skipped_patient_imgs = np.zeros((self.num_frames_batch, self.img_size_load[0], self.img_size_load[1], 1))
        for i in range(self.num_frames_batch):
            skipped_patient_imgs[i] = patient_imgs[i*frame_skip]    
        return skipped_patient_imgs
    
    # Select a random window of patient frames, in case its images has more frames than self.num_frame_batch 
    
    def randomWindow(self, patient_imgs):
        windowed_imgs = np.zeros((self.num_frames_batch, patient_imgs.shape[1], patient_imgs.shape[2], 1))
        num_frames = patient_imgs.shape[0]
        if num_frames < self.num_frames_batch:
            windowed_imgs[:num_frames] = patient_imgs
        else:
            random_frames = np.arange(num_frames)
            index = np.random.randint(0, num_frames - self.num_frames_batch)
            windowed_imgs[0:] = patient_imgs[index:index+self.num_frames_batch]
        return windowed_imgs
            
    
    # Convert raw frames to a fix size array -> (batch_size, num_frames_batch, img_size_crop[0], img_size_crop[1], 1)
    
    def groupImages(self, list_scan_imgs):
        grouped_imgs = []
        for patient_imgs in list_scan_imgs:
            if patient_imgs.shape[1] > self.num_frames_batch:
                patient_imgs = self.filterSlices(patient_imgs)
            if self.random_window:
                patient_imgs = self.randomWindow(patient_imgs)
            else:
                patient_imgs = self.frameSkipImages(patient_imgs)
            grouped_imgs.append(patient_imgs)
        return np.asarray(grouped_imgs)
        
    # Performs augmentation operations conserving the 3D property on the z axis
    
    def loadImagesAugmented(self, patient_imgs):

        if self.img_size_load != self.img_size_crop:
            patient_imgs = self.center3Dcropping(patient_imgs)
            if patient_imgs.shape[2] > self.img_size_crop[0] and patient_imgs.shape[3] > self.img_size_crop[1]:
                patient_imgs = self.random3DCropping(patient_imgs)
        if self.training and np.random.random() > 0.5:
            patient_imgs = np.fliplr(patient_imgs)
        if self.training and np.random.random() > 0.5:
            patient_imgs = np.flipud(patient_imgs)
        if self.training and np.random.random() > 0.5:
            patient_imgs = patient_imgs[:, :, ::-1]
        if self.training and np.random.random() > 0.5:
            patient_imgs = patient_imgs[:, ::-1, :]
        if self.training:
            patient_rotated_imgs= []
            for batch in range(patient_imgs.shape[0]):
                angle = np.random.randint(-15, 15)
                batch_imgs_rotated = np.asarray([ndimage.rotate(patient_imgs[batch, i], angle, order=1,
                                                                reshape=False) for i in range(patient_imgs.shape[1])])
                patient_rotated_imgs.append(batch_imgs_rotated)
            patient_imgs = np.asarray(patient_rotated_imgs) 
        return patient_imgs
    
    
    #Random Cropping 3D - change x, y axis but not z
    
    def random3DCropping(self, patient_imgs):
        w, h = self.img_size_crop[0], self.img_size_crop[1]
        x = np.random.randint(0, patient_imgs.shape[2] - w)
        y = np.random.randint(0, patient_imgs.shape[2] - h)
        patient_crop_imgs = patient_imgs[:, :, y:y+h, x:x+w]
        return patient_crop_imgs
    
    # Center 3D Cropping
    
    def center3Dcropping(self, patient_imgs):
        w, h = patient_imgs.shape[2] - 20, patient_imgs.shape[3] - 20
        img_height, img_width = patient_imgs.shape[2], patient_imgs.shape[3]
        left, right = (img_width - w) / 2, (img_width + w) / 2
        top, bottom = (img_height - h) / 2, (img_height + h) / 2
        left, top = round(max(0, left)), round(max(0, top))
        right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
        patient_crop_imgs = patient_imgs[:, :, top:bottom, left:right]
        return patient_crop_imgs
    
    # We shuffle the data at the end of each epoch
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        np.random.shuffle(self.indexes)
     
    # Get only one patient, for debugging or prediction
        
    def getOnePatient(self, patient_id):
        list_scan_imgs = [decodePatientImages(patient_id,
                                              self.dict_patients_masks_paths,
                                              image_size=(self.img_size_load[0], self.img_size_load[1]),
                                              numpy=True)] 
        patient_imgs = self.groupImages(list_scan_imgs)
        patient_imgs = self.loadImagesAugmented(patient_imgs)
        self.dict_ini_features[patient_id]['Patient'] = patient_id
        return (patient_imgs, [self.dict_ini_features[patient_id]])


## Model 

## 05. Models

### 05.1 Backbone 3D Image Model

class BackBone3DModel(models.Model):
    
    def __init__(self, unet=False, path_unet=None):
        super(BackBone3DModel, self).__init__(name='BackBone3DModel')
        self.unet = unet
        self.path_unet = path_unet
        if self.unet:
            self.unet_model = tf.keras.models.load_model(self.path_unet, compile=False) 
            self.unet_model.trainable = False
        else:
            self.avg_pool = layers.AvgPool3D(pool_size=(2, 1, 1), name='avg_pool')
            
            self.block1_conv1 = layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block1_conv1')
            self.block1_conv2 = layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block1_conv2')
            self.block1_maxpool1 = layers.MaxPool3D(pool_size=(2, 2, 2), name='block1_maxpool1')

            self.block2_conv1 = layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block2_conv1')
            self.block2_conv2 = layers.Conv3D(128, kernel_size=(3, 3, 3),padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block2_conv2')
            self.block2_maxpool1 = layers.MaxPool3D(pool_size=(2, 2, 2), name='block2_maxpool1')

            self.block3_conv1 = layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block3_conv1')
            self.block3_conv2 = layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block3_conv2')
            self.block3_maxpool1 = layers.MaxPool3D(pool_size=(2, 2, 2), name='block3_maxpool1')
            
            self.block4_conv1 = layers.Conv3D(512, kernel_size=(3, 3, 3), padding='same', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              activation='relu', name='block3_conv2')
        
    
    def call(self, inputs):
        if self.unet:
            x = self.unet_model(inputs)
            # (None, 2, 20, 20, 256)  
        else:
            x = self.avg_pool(inputs)
            x = self.block1_conv1(x)
            x = self.block1_conv2(x)
            x = self.block1_maxpool1(x)

            x = self.block2_conv1(x)
            x = self.block2_conv2(x)
            x = self.block2_maxpool1(x)

            x = self.block3_conv1(x)
            x = self.block3_conv2(x)
            x = self.block3_maxpool1(x)
            
            x = self.block4_conv1(x)
        
        return x
    

### 05.2 Backbone Tabular Data for Patients metadata

class BackBoneTabularModel(models.Model):
    
    def __init__(self, dense_dim, dropout_rate, sex_dim=20, smoker_dim=20, max_norm=1):
        super(BackBoneTabularModel, self).__init__(name='BackBoneTabularModel')
        
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
 
        self.sex_dim = sex_dim 
        self.smoker_dim = smoker_dim
        
        # Embedding layers
        self.emb_sex = layers.Embedding(input_dim=2, output_dim=self.sex_dim)
        self.emb_smoker = layers.Embedding(input_dim=3, output_dim=self.smoker_dim)
        
        # Output layer
        self.dropout = layers.Dropout(self.dropout_rate)
        self.dense = layers.Dense(units=self.dense_dim, 
                                  kernel_regularizer=regularizers.l2(1e-4),
                                  kernel_constraint = constraints.MaxNorm(max_norm),
                                  name='tabular_dense')
        self.batch_norm = layers.BatchNormalization()
        
        
    def call(self, inputs, training=True):

        patient_sex = self.emb_sex(inputs[:, 0])
        patient_smoke = self.emb_smoker(inputs[:, 1])
        x = tf.concat([patient_sex,
                       patient_smoke,
                       tf.expand_dims(inputs[:, 2], 1), #Age
                       tf.expand_dims(inputs[:, 3], 1)], axis=-1) # Percent
        
        x = self.dropout(x, training)
        x = self.dense(x)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        
        return x
        
                      
### 05.3 Encoder Model

class Encoder(models.Model):
    
    def __init__(self, features_dim, dropout_rate=0.2, unet=False, path_unet=None,
                       tabular_dense_dim=16, tabular_dropout_rate=0.4,
                       tabular_sex_dim=10, tabular_smoker_dim=10, max_norm=1, recurrent_max_norm=0.1,
                     **kwargs):
                    
        super(Encoder, self).__init__( **kwargs, name='Encoder')
        self.backbone_img_model = BackBone3DModel(unet, path_unet)
        self.backbone_tabular_model = BackBoneTabularModel(dense_dim=tabular_dense_dim, 
                                                           dropout_rate=tabular_dropout_rate,  
                                                           sex_dim=tabular_sex_dim, 
                                                           smoker_dim=tabular_smoker_dim,
                                                           max_norm=max_norm)
        
        self.dropout = layers.Dropout(dropout_rate, name='dropout')
        self.dense = layers.Dense(features_dim, name='encoder_dense', 
                                              kernel_regularizer=regularizers.l2(1e-4),
                                              kernel_constraint=constraints.MaxNorm(max_norm))
        self.batch_norm2 = layers.BatchNormalization()

    
    def call(self, img_inputs, scalar_inputs, training=True):
        # Image Features from 3D Model
        img_features = self.backbone_img_model(img_inputs)
        img_dim = img_features.shape[1]*img_features.shape[2]*img_features.shape[3]
        img_features = tf.reshape(img_features, tf.convert_to_tensor([img_features.shape[0], 
                                                                     img_dim, 
                                                                     img_features.shape[4]], dtype=tf.int32))
        
        
        # Scalar Features from Patient Metadata
        scalar_features = self.backbone_tabular_model(scalar_inputs, training)
        self.repeatvector = layers.RepeatVector(img_features.shape[1])
        scalar_features = self.repeatvector(scalar_features)
               
        # Mixing both together
        features = tf.concat(values=[img_features, scalar_features], axis=-1)
        features = self.dropout(features, training=training)
        features = self.dense(features)
        feaures = self.batch_norm2(features)
        features = tf.nn.relu(features)
        
        return features
    

### 05.4 Decoder Model

class Decoder(models.Model):
    
    def __init__(self, embedding_dim, rnn_units=[64], dense_units=[64], dense_activation=True,
                 dropout_rate=0.2, max_norm=1, recurrent_max_norm=1,
                 attention='bahdanau', **kwargs):
        super(Decoder, self).__init__(**kwargs, name='Decoder')
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.attention_features_shape = self.rnn_units[-1]
        self.dropout_rate = dropout_rate
        self.attention = attention.lower()
        self.max_norm = max_norm
        self.recurrent_max_norm = recurrent_max_norm

        if attention.lower() == 'bahdanau':
            self.attention_layer = BahdanauAttention(self.attention_features_shape)   
        elif attention.lower() == 'dotproduct':
            self.attention_layer = ScaledDotProductAttention(self.attention_features_shape)   
        elif attention.lower() == 'general':
            self.attention_layer = GeneralAttention(self.attention_features_shape)   
        else:
            raise ValueError(f'Attention {attention} not valid. Choose between bahdanau/dotproduct/general.')
            
        self.flatten = layers.Flatten(name='flatten')
        self.dropout = layers.Dropout(self.dropout_rate)
        self.dense_activation = dense_activation
        self.grus = self.stackRNN()
        if self.dense_units:
            self.fcc_denses = self.stackDense()
        
        self.dense_output = layers.Dense(1, activation='linear')
        self.dense_distribution = layers.Dense(2, activation='linear')
        
        self.distribution = tfp.layers.DistributionLambda(
                                lambda t: tfd.Laplace(loc=t[:, 0],
                                                     scale=0.01*tf.math.softplus(t[:, 1])),
                                name='laplace_dist')
    
    def call(self, decoder_input, features, hidden, training=True):
        context_vector, attention_weights = self.attention_layer(features, hidden)
        
        x = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(decoder_input, 1)], axis=-1)
            
        for gru in self.grus[:-1]:
            x = gru(x, training=training)
            
        output, state = self.grus[-1](x, training=training)        
        x = self.flatten(output)
        
        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)
            
        if self.dense_units:
            for fcc in self.fcc_denses:
                x = fcc(x)
        
        x_output = self.dense_output(x)
        
        x_distribution = self.dense_distribution(x)
        x_distribution = self.distribution(x_distribution)
        
        outputs = (x_output, x_distribution)
        
        return outputs, state, attention_weights
    
    def stackRNN(self):
        rnns = []
        for units in self.rnn_units[:-1]:
            gru_ = layers.GRU(units, 
                              return_state=False,
                              return_sequences=True, 
                              kernel_constraint=constraints.MaxNorm(self.max_norm),
                              recurrent_constraint=constraints.MaxNorm(self.recurrent_max_norm),
                              kernel_regularizer=regularizers.l2(1e-4),
                              recurrent_initializer='glorot_uniform')
            rnns.append(gru_)
        
        gru_ = layers.GRU(self.rnn_units[-1], 
                          return_sequences=True, 
                          return_state=True,
                          kernel_constraint=constraints.MaxNorm(self.max_norm),
                          recurrent_constraint=constraints.MaxNorm(self.recurrent_max_norm),
                          kernel_regularizer=regularizers.l2(1e-4),
                          recurrent_initializer='glorot_uniform')
        
        rnns.append(gru_)
        return rnns
    
    
    def stackDense(self):
        denses, batch_norms = [], []
        for units in self.dense_units:
            dense_ = layers.Dense(units,
                                  activation=self.dense_activation,
                                  kernel_constraint=constraints.MaxNorm(self.max_norm),
                                  kernel_regularizer=regularizers.l2(1e-4))
            denses.append(dense_)
        return denses
    
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.rnn_units[0]))


### 05.5 Global Model
    
class PulmonarFibrosisEncoderDecoder(models.Model):
    
    def __init__(self, encoder_tabular_dense_dim, encoder_tabular_dropout_rate, 
                 encoder_tabular_sex_dim, encoder_tabular_smoker_dim, encoder_feature_dim, encoder_unet, encoder_path_unet,
                 encoder_max_norm, encoder_dropout_rate, decoder_embedding_dim, decoder_rnn_units, 
                 decoder_dense_units, decoder_dense_activation, decoder_attention,
                 decoder_dropout_rate, decoder_max_norm, decoder_recurrent_max_norm,
                 learning_rate, clipvalue, checkpoint_path, teacher_forcing, batch_size,
                 learning_rate_epoch_decay, epsilon, epsilon_decay, save_checkpoints, 
                 restore_last_checkpoint, dict_train_sequence_fvc, dict_train_sequence_weekssincelastvisit,
                 dict_train_patients_masks_paths, dict_patients_train_ini_features, mean_fvc, std_fvc,
                 **kwargs):
        
        super(PulmonarFibrosisEncoderDecoder, self).__init__(**kwargs, name='PulmonarFibrosisEncoderDecoder')
        
        tf.keras.backend.clear_session()
        
        # Global dicts
        self.dict_train_sequence_fvc = dict_train_sequence_fvc
        self.dict_train_sequence_weekssincelastvisit = dict_train_sequence_weekssincelastvisit
        self.dict_train_patients_masks_paths = dict_train_patients_masks_paths
        self.dict_patients_train_ini_features = dict_patients_train_ini_features
        
        # Encoder
        self.encoder_tabular_dense_dim = encoder_tabular_dense_dim
        self.encoder_tabular_sex_dim = encoder_tabular_sex_dim
        self.encoder_tabular_smoker_dim = encoder_tabular_smoker_dim
        self.encoder_tabular_dropout_rate = encoder_tabular_dropout_rate
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder_path_unet = encoder_path_unet
        self.unet = encoder_unet
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_max_norm = encoder_max_norm
        
        # Decoder
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_rnn_units = decoder_rnn_units
        self.decoder_dense_units = decoder_dense_units
        self.decoder_dropout_rate = decoder_dropout_rate
        self.decoder_dense_activation = decoder_dense_activation
        self.decoder_attention = decoder_attention
        self.decoder_max_norm = decoder_max_norm
        self.decoder_recurrent_max_norm = decoder_recurrent_max_norm
        
        # Utils - Training 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clipvalue = clipvalue
        self.teacher_forcing = teacher_forcing
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
        self.epsilon_decay = tf.convert_to_tensor(epsilon_decay, dtype=tf.float32)

        # Build model
        self.learning_rate_epoch_decay = learning_rate_epoch_decay
        self.build()
        self.compile()
        
        # Utils
        self.save_checkpoints = save_checkpoints
        self.checkpoint_path = checkpoint_path
        self.buildCheckpoints()
        if restore_last_checkpoint:
            self.ckpt.restore(sorted(self.ckpt_manager.checkpoints)[-1])
        self.mean_fvc=mean_fvc
        self.std_fvc=std_fvc
                
                
    def build(self):
        self.encoder = Encoder(features_dim=self.encoder_feature_dim,
                               tabular_sex_dim=self.encoder_tabular_sex_dim, 
                               tabular_smoker_dim=self.encoder_tabular_smoker_dim,
                               tabular_dense_dim=self.encoder_tabular_dense_dim, 
                               tabular_dropout_rate=self.encoder_tabular_dropout_rate,
                               dropout_rate=self.encoder_dropout_rate, 
                               unet=self.unet, path_unet=self.encoder_path_unet,
                               max_norm=self.encoder_max_norm)
        
        self.decoder = Decoder(embedding_dim=self.decoder_embedding_dim, 
                               rnn_units=self.decoder_rnn_units, 
                               dense_units=self.decoder_dense_units, dropout_rate=self.decoder_dropout_rate,
                               dense_activation=self.decoder_dense_activation,
                               attention=self.decoder_attention, max_norm=self.decoder_max_norm,
                               recurrent_max_norm=self.decoder_recurrent_max_norm)
        
        
    def compile(self):
        super(PulmonarFibrosisEncoderDecoder, self).compile()
            
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipvalue=self.clipvalue)
        self.loss_function = customLossFunction
        self.metric = [tf.keras.losses.MeanSquaredError(name='mse')]
        
   
    def buildCheckpoints(self):
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
            
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                        decoder=self.decoder,
                                        optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)
                               

    @tf.function
    def trainStep(self, img_tensor, features_tensor, weeks_since_lastvisit_tensor, initial_fvc, target):
        loss, metric = 0, 0
        list_predictions = []
        hidden = self.decoder.reset_state(batch_size=self.batch_size)
        decoder_input = tf.convert_to_tensor([[initial_fvc[0, 0], weeks_since_lastvisit_tensor[0, 0]]], dtype=np.float32)
        
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor, 
                                    features_tensor,
                                    training=True)
            for i in range(0, weeks_since_lastvisit_tensor.shape[0]):
                predictions, hidden, _ = self.decoder(decoder_input, 
                                                      features, 
                                                      hidden,
                                                      training=True)
                
                pred_output, pred_distribution = predictions
                pred_std = tf.clip_by_value(pred_distribution.stddev() * 100, 0., 100.)
                pred_mean = pred_distribution.mean()

                loss += self.loss_function(unscale(target[:, i], self.mean_fvc, self.std_fvc),
                                           unscale(pred_mean, self.mean_fvc, self.std_fvc), std=pred_std)
                
                metric += self.metric[0](target[:, i], pred_mean)
                
                # Teacher forcing
                if self.teacher_forcing=='avg':
                    teacher_forc = tf.expand_dims(tf.reduce_mean([target[:, i], pred_mean]), 0)
                elif self.teacher_forcing=='random':
                    random_ = np.random.random()
                    if random_ > 0.5:
                        teacher_forc = target[:, i]
                    else:
                        teacher_forc = pred_mean # pred_output[0]
                else:
                    teacher_forc = (target[:, i] * self.epsilon) + (pred_mean * (1-self.epsilon))
                    
                list_predictions.append(pred_mean)
                if i <= weeks_since_lastvisit_tensor.shape[0]:
                    decoder_input = tf.expand_dims(tf.concat([teacher_forc, weeks_since_lastvisit_tensor[i]], axis=-1), 0)
                
        
        total_loss = (loss/int(len(target)))
        total_metric = (metric/int(len(target)))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        return loss, total_loss, list_predictions, total_metric
    
    
    def fitModel(self, X_train, X_val=None, epochs=1):
        history = {}
        history['loss'], history['val_loss'], history['metric'] = [], [], []
    
        for epoch in range(0, epochs):
            start = time.time()
            print(f'Epoch [{epoch+1}/{epochs}]')
            len_X_val = 0 if X_val is None else len(X_val)
            len_X_train = len(X_train)
            pbar = tf.keras.utils.Progbar(len_X_train + len_X_val)
            total_loss, total_metric = 0, 0

            # Train
            
            for num_batch, batch in enumerate(X_train):
                img_tensor, features_tensor = batch[0], batch[1]
                patients = [dict_['Patient'] for dict_ in features_tensor]
                target_original = [self.dict_train_sequence_fvc[patient] for patient in patients]
                initial_fvc = [self.dict_patients_train_ini_features[patient]['FVC'] for patient in patients]
                target = tf.convert_to_tensor(target_original, dtype=np.float32)
                features_tensor = tf.convert_to_tensor([[p['Sex'], 
                                                         p['SmokingStatus'],
                                                         p['Age'],
                                                         p['Percent']] for p in features_tensor], dtype=tf.float32)
                    
                weeks_since_lastvisit_tensor = tf.convert_to_tensor(
                                                     [self.dict_train_sequence_weekssincelastvisit[patient] for patient in patients], 
                                                    dtype=tf.float32)
                weeks_since_lastvisit_tensor = tf.reshape(weeks_since_lastvisit_tensor, 
                                                          [weeks_since_lastvisit_tensor.shape[1], 1])

                batch_loss, batch_loss, list_predictions, batch_metric = self.trainStep(img_tensor, 
                                                                                        features_tensor, 
                                                                                        weeks_since_lastvisit_tensor,
                                                                                        tf.convert_to_tensor([initial_fvc],dtype=tf.float32),
                                                                                        target)
                
                total_loss += batch_loss
                total_metric += batch_metric

                pbar.update(num_batch + 1, values=[('Loss', batch_loss)] + [('mse', batch_metric)])
                
            self.epsilon = self.epsilon * self.epsilon_decay
            total_loss  /= float(len_X_train)
            total_metric  /= float(len_X_train)
            history['loss'].append(total_loss)
            history['metric'].append(total_metric)
            
            # Validation
            if X_val:
                val_total_loss = 0
                for num_batch, batch in enumerate(X_val):
                    img_tensor, features_tensor = batch[0], batch[1]
                    patients = [dict_['Patient'] for dict_ in features_tensor]
                    target_original = [self.dict_train_sequence_fvc[patient] for patient in patients]
                    initial_fvc = [self.dict_patients_train_ini_features[patient]['FVC'] for patient in patients]
                    target = tf.convert_to_tensor(target_original, dtype=np.float32)
                    features_tensor = tf.convert_to_tensor([[p['Sex'], 
                                                             p['SmokingStatus'],
                                                             p['Age'],
                                                             p['Percent']] for p in features_tensor], dtype=tf.float32)

                    weeks_since_lastvisit_tensor = tf.convert_to_tensor(
                                                     [self.dict_train_sequence_weekssincelastvisit[patient] for patient in patients], 
                                                    dtype=tf.float32)
                    weeks_since_lastvisit_tensor = tf.reshape(weeks_since_lastvisit_tensor, 
                                                          [weeks_since_lastvisit_tensor.shape[1], 1])
                     
                    dict_output_tensors = self.predictStep(img_tensor, 
                                                            features_tensor, 
                                                            weeks_since_lastvisit_tensor, 
                                                            tf.convert_to_tensor([initial_fvc],dtype=tf.float32))
    
                    predictions = dict_output_tensors['predictions']
                    confidences = dict_output_tensors['confidence']
                    val_batch_loss = self.loss_function(unscale(target, self.mean_fvc, self.std_fvc), 
                                                        unscale(predictions, self.mean_fvc, self.std_fvc), 
                                                        std=confidences[:])
                    val_metric = self.metric[0](target, predictions)
                    val_total_loss += val_batch_loss
                    
                    pbar.update(len_X_train + num_batch + 1, values=[('val_loss', val_batch_loss)] + [('val_mse', val_metric)])
                    
                val_total_loss  /= float(len_X_val)
                history['val_loss'].append(val_total_loss)
            
            if self.learning_rate_epoch_decay > 0  and self.learning_rate_epoch_decay < 1:
                self.optimizer.learning_rate = self.optimizer.learning_rate * self.learning_rate_epoch_decay
            X_train.on_epoch_end() 
            if X_val:
                X_val.on_epoch_end()
            if self.save_checkpoints:
                self.ckpt_manager.save()
            print(' ({:.0f} sec)\n'.format( time.time() - start))
            
        return history
    
    
    @tf.function
    def predictStep(self, img_tensor, features_tensor, weeks_since_lastvisit_tensor, initial_fvc):
        
        output_tensors = {}
        list_predictions, list_condidences = [], []
        hidden = self.decoder.reset_state(batch_size=1)
        decoder_input = tf.convert_to_tensor([[initial_fvc[0, 0], weeks_since_lastvisit_tensor[0, 0]]], dtype=tf.float32)

        encoder_features_tensor = self.encoder(img_tensor, 
                                               features_tensor,
                                               training=False)
        attention_plot = []
        for i in range(0, weeks_since_lastvisit_tensor.shape[0]):
            predictions, hidden, attention_weights = self.decoder(decoder_input, 
                                                                  encoder_features_tensor, 
                                                                  hidden,
                                                                  training=False)
            pred_output, pred_distribution = predictions
            pred_std = tf.clip_by_value(pred_distribution.stddev() * 100, 0., 100.)
            pred_mean = pred_distribution.mean()
                            
            attention_plot.append(tf.reshape(attention_weights, (-1, ))) 
            list_predictions.append(pred_mean)
            list_condidences.append(pred_std)
            
            if i <= weeks_since_lastvisit_tensor.shape[0]:
                decoder_input = tf.expand_dims(tf.concat(values=[pred_mean,
                                                                 weeks_since_lastvisit_tensor[i]], axis=-1), 0)    
    
            
        output_tensors['predictions'] = tf.convert_to_tensor(list_predictions, dtype=tf.float32)
        confidences_tensor = tf.convert_to_tensor(list_condidences, dtype=tf.float32)
        output_tensors['confidence'] = tf.reshape(confidences_tensor, (confidences_tensor.shape[0],
                                                                       confidences_tensor.shape[1]))
        output_tensors['attention_plot'] = tf.convert_to_tensor(attention_plot, dtype=tf.float32)

        return output_tensors
    
        
    def predictEvaluateModel(self, X_generator, patient, initial_fvc, list_weeks_elapsed, batch=None):

        if not batch:
            batch = X_generator.getOnePatient(patient)
        img_tensor, features_tensor = batch[0], batch[1]
        
        features_tensor = tf.convert_to_tensor([[p['Sex'], 
                                                 p['SmokingStatus'],
                                                 p['Age'],
                                                 p['Percent']] for p in features_tensor], dtype=tf.float32)
        
        weeks_since_lastvisit_tensor = tf.convert_to_tensor([list_weeks_elapsed], dtype=tf.float32)
        weeks_since_lastvisit_tensor = tf.reshape(weeks_since_lastvisit_tensor, [weeks_since_lastvisit_tensor.shape[1], 1])
        initial_fvc = [self.dict_train_sequence_fvc[patient][0]]
         
        dict_output_tensors = self.predictStep(img_tensor, 
                                                features_tensor, 
                                                weeks_since_lastvisit_tensor, 
                                                tf.convert_to_tensor([initial_fvc], dtype=tf.float32))
        
        list_predictions, attention_plot = dict_output_tensors['predictions'], dict_output_tensors['attention_plot']
        list_confidences = dict_output_tensors['confidence']
        attention_plot = attention_plot[:len(list_predictions)]
        
        return np.asarray(list_predictions),  np.asarray(list_confidences), np.asarray(attention_plot)
  


# Display Attention Plot

def plotAttention(images, list_weeks_elapsed, result, attention_plot, alpha=0.7, max_imgs=False):
        
        fig = plt.figure(figsize=(10, 10))
        if max_imgs:
            temp_image = np.max(images, axis=0)
        len_result = len(result)
        for i in range(len_result):
            if not max_imgs:
                temp_image = images[i]
            temp_att = np.resize(attention_plot[i], (8, 8))
            if len_result >= 2:
                ax = fig.add_subplot(len_result//2, len_result//2, i+1)
            else:
                ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f'Weeks: {list_weeks_elapsed[i]} - Pred: {int(result[i])}')
            img = ax.imshow(temp_image, cmap=plt.cm.bone)
            ax.imshow(temp_att, cmap='gray', alpha=alpha, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

