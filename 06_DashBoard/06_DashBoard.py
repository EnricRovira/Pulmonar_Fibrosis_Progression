#!/usr/bin/env python
# coding: utf-8

# # Fibrosis Progression DashBoard 

# ## 1. Libraries

#####################################################################
# 01. Libraries

# streamlit run 06_DashBoard.py
import os
import pandas as pd
import numpy as np
import random
np.random.seed(12)

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt
import base64
import glob
import cv2

import time
from tqdm import tqdm

import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation, color
import pydicom
import imageio

from utils_dashboard import loadSlices, getPixelsHu, plotHistogramPixelesHu, plotSampleStack, buildWindowGif, PulmonarFibrosisEncoderDecoder, SequenceToSequenceDataGenerator, plotAttention
from utils import *
import streamlit as st
import ast

#####################################################################

#####################################################################
# 02. Global Variables

path = '../01_Data/'

path_imgs_train = path + 'train_imgs/'
path_imgs_test = path + 'test_imgs/'

path_masks_train = path + 'train_masks/'
path_masks_test = path + 'test_masks/'

path_models = '../05_Saved_Models/'

#####################################################################


#########################################################################
# 03. Load Data & Preprocess Data

@st.cache(allow_output_mutation=True)
def loadPreprocessData():
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')

    print(f'1.1 -> There are {df_train.Patient.unique().shape[0]} train unique patients')
    print(f'1.2 -> There are {df_test.Patient.unique().shape[0]} test unique patients')

    train_mask_paths = glob.glob(path_masks_train + '*')
    test_mask_paths = glob.glob(path_masks_test + '*')

    print(f'No. of Train Masks : {len(train_mask_paths)}')
    print(f'No. of Test Masks : {len(test_mask_paths)}')
        
    unique_train_patients = df_train.Patient.unique()
    unique_test_patients = df_test.Patient.unique()

    dict_train_patients_paths = {patient: path_imgs_train + patient + '/' for patient in unique_train_patients}
    dict_test_patients_paths = {patient: path_imgs_test + patient + '/' for patient in unique_test_patients}

    dict_train_patients_masks_paths = {patient: path_masks_train + patient + '/' for patient in unique_train_patients}
    dict_test_patients_masks_paths = {patient: path_masks_test + patient + '/' for patient in unique_test_patients}

    for patient in tqdm(dict_train_patients_paths):
        list_files = os.listdir(dict_train_patients_paths[patient])
        list_files = [dict_train_patients_paths[patient] + file for file in list_files]
        dict_train_patients_paths[patient] = list_files
        
    for patient in tqdm(dict_test_patients_paths):
        list_files = os.listdir(dict_test_patients_paths[patient])
        list_files = [dict_test_patients_paths[patient] + file for file in list_files]
        dict_test_patients_paths[patient] = list_files

        
    for patient in tqdm(dict_train_patients_masks_paths):
        if os.path.exists(dict_train_patients_masks_paths[patient]):
            list_files = os.listdir(dict_train_patients_masks_paths[patient])
            list_files = [dict_train_patients_masks_paths[patient] + file for file in list_files]
            dict_train_patients_masks_paths[patient] = list_files
        
    for patient in tqdm(dict_test_patients_masks_paths):
        list_files = os.listdir(dict_test_patients_masks_paths[patient])
        list_files = [dict_test_patients_masks_paths[patient] + file for file in list_files]
        dict_test_patients_masks_paths[patient] = list_files

    # Preprocessing:

    df_train = df_train.groupby(['Patient', 'Weeks']).agg({
        'FVC': np.mean,
        'Percent': np.mean,
        'Age': np.max,
        'Sex': np.max,
        'SmokingStatus': np.max 
    }).reset_index()

    # Noramlize fvc

    mean_fvc, std_fvc = df_train.FVC.mean(), df_train.FVC.std()
    mean_perc, std_perc = df_train.Percent.mean(), df_train.Percent.std()
    mean_age, std_age = df_train.Age.min(), df_train.Age.max()

    df_train['Age'] = df_train['Age'].apply(lambda x: (x-mean_age)/std_age)
    df_test['Age'] = df_test['Age'].apply(lambda x: (x-mean_age)/std_age)

    df_train['FVC'] = df_train['FVC'].apply(lambda x: (x-mean_fvc)/std_fvc)
    df_test['FVC'] = df_test['FVC'].apply(lambda x: (x-mean_fvc)/std_fvc)

    df_train['Percent'] = df_train['Percent'].apply(lambda x: (x-mean_perc)/std_perc)
    df_test['Percent'] = df_test['Percent'].apply(lambda x: (x-mean_perc)/std_perc)

    # We will consider first week as ct-scan week

    df_train['Weeks'] = df_train['Weeks'].apply(lambda x: x if x>=0 else 0)
    df_test['Weeks'] = df_test['Weeks'].apply(lambda x: x if x>=0 else 0)

    df_train = df_train.sort_values(['Patient', 'Weeks']).reset_index(drop=True)
    df_test = df_test.sort_values(['Patient', 'Weeks']).reset_index(drop=True)

    df_train['ElapsedWeeks'] = df_train['Weeks']
    df_test['ElapsedWeeks'] = df_test['Weeks']

    train_weeks_elapsed = df_train.set_index(['Patient', 'Weeks'])['ElapsedWeeks'].diff().reset_index()
    test_weeks_elapsed = df_test.set_index(['Patient', 'Weeks'])['ElapsedWeeks'].diff().reset_index()

    df_train = df_train.drop('ElapsedWeeks', axis=1)
    df_test = df_test.drop('ElapsedWeeks', axis=1)

    train_weeks_elapsed['ElapsedWeeks'] = train_weeks_elapsed['ElapsedWeeks'].fillna(0).astype(int)
    test_weeks_elapsed['ElapsedWeeks'] = test_weeks_elapsed['ElapsedWeeks'].fillna(0).astype(int)

    train_weeks_elapsed['ElapsedWeeks'] = train_weeks_elapsed['ElapsedWeeks'].apply(lambda x: 0 if x<0 else x)
    test_weeks_elapsed['ElapsedWeeks'] = test_weeks_elapsed['ElapsedWeeks'].apply(lambda x: 0 if x<0 else x)

    df_train = df_train.merge(train_weeks_elapsed, how='inner', on=['Patient', 'Weeks'])
    df_test = df_test.merge(test_weeks_elapsed, how='inner', on=['Patient', 'Weeks'])

    df_train['patient_row'] = df_train.sort_values(['Patient', 'Weeks'], ascending=[True, True]) \
                .groupby(['Patient']) \
                .cumcount() + 1

    df_test['patient_row'] = df_test.sort_values(['Patient', 'Weeks'], ascending=[True, True]) \
                .groupby(['Patient']) \
                .cumcount() + 1


    df_train['WeeksSinceLastVisit'] = df_train.apply(lambda x: x['Weeks'] if x['patient_row']==1 else x['ElapsedWeeks'], axis=1)
    df_test['WeeksSinceLastVisit'] = df_test.apply(lambda x: x['Weeks'] if x['patient_row']==1 else x['ElapsedWeeks'], axis=1)

    # Ini dictionaries

    columns = ['FVC', 'Age', 'Sex', 'SmokingStatus', 'WeeksSinceLastVisit', 'Percent']
    dict_patients_train_ini_features, dict_patients_test_ini_features = {}, {}
    df_train_patients, df_test_patients = df_train.set_index('Patient'), df_test.set_index('Patient')

    # Mapping categories dictionaries 
    dict_patients = {k: v for k, v in enumerate(unique_train_patients)}
    dict_patients_inv = {v: k for k, v in enumerate(unique_train_patients)}

    dict_sex = {'Male': 0, 'Female': 1}
    dict_sex_inv = {0: 'Male', 1: 'Female'}

    dict_smoke = {'Ex-smoker': 0, 'Never smoked': 1, 'Currently smokes': 2}
    dict_smoke_inv = {0: 'Ex-smoker', 1:'Never smoked', 2:'Currently smokes'}

    df_train_patients.index = df_train_patients.index.to_series().map(dict_patients_inv).values
    df_train_patients.Sex = df_train_patients.Sex.apply(lambda x: dict_sex[x])
    df_train_patients.SmokingStatus = df_train_patients.SmokingStatus.apply(lambda x: dict_smoke[x])

    df_test_patients.index = df_test_patients.index.to_series().map(dict_patients_inv).values
    df_test_patients.Sex = df_test_patients.Sex.apply(lambda x: dict_sex[x])
    df_test_patients.SmokingStatus = df_test_patients.SmokingStatus.apply(lambda x: dict_smoke[x])

    for patient in unique_train_patients:
        dict_patients_train_ini_features[patient] = \
            df_train_patients[columns].loc[dict_patients_inv[patient], :].to_dict('records')[0]
        
    for patient in unique_test_patients:
        dict_patients_test_ini_features[patient] = \
            df_test_patients[columns].loc[dict_patients_inv[patient], :].to_dict()

    # Decoder inputs
    dict_train_sequence_fvc, dict_train_sequence_weekssincelastvisit = {}, {}
    for patient in unique_train_patients:
        dict_train_sequence_fvc[patient] = list(df_train_patients['FVC'].loc[dict_patients_inv[patient]].values[1:])
        dict_train_sequence_weekssincelastvisit[patient] = list(df_train_patients['WeeksSinceLastVisit'].loc[dict_patients_inv[patient]].values[1:])

    return dict_train_sequence_weekssincelastvisit, dict_patients_train_ini_features, dict_train_sequence_fvc, mean_fvc, std_fvc, dict_train_patients_masks_paths, dict_train_patients_paths, unique_test_patients, df_train, dict_sex_inv, dict_smoke_inv, mean_age, std_age, mean_perc, std_perc

dict_train_sequence_weekssincelastvisit, dict_patients_train_ini_features, dict_train_sequence_fvc, mean_fvc, std_fvc, dict_train_patients_masks_paths, dict_train_patients_paths, unique_test_patients, df_train, dict_sex_inv, dict_smoke_inv, mean_age, std_age, mean_perc, std_perc = loadPreprocessData()

#####################################################################


#####################################################################
# 04. Import Model

model_inputs = dict(
    # Encoder
    encoder_tabular_dense_dim=32, 
    encoder_tabular_dropout_rate=0.4,
    encoder_tabular_sex_dim=20, 
    encoder_tabular_smoker_dim=20,
    encoder_feature_dim = 256,
    encoder_unet=True,
    encoder_path_unet=path_models + 'encoder_unet3d.h5',
    encoder_dropout_rate=0.4,
    encoder_max_norm=0.1,
    # Decoder
    decoder_embedding_dim = 256, 
    decoder_rnn_units = [256], 
    decoder_dense_units = [],
    decoder_dense_activation=None,
    decoder_dropout_rate = 0.4,
    decoder_max_norm=0.1,
    decoder_recurrent_max_norm=0.1,
    decoder_attention = 'bahdanau',
    # Training
    learning_rate = 2e-4,
    clipvalue=0.5,
    teacher_forcing = 'decay', # avg/random/decay
    batch_size=1, 
    epsilon=1.0, 
    epsilon_decay=1.0,
    # Utils
    learning_rate_epoch_decay=0.85,
    checkpoint_path=path_models + 'checkpoints_no_gpu/',
    save_checkpoints=True,
    restore_last_checkpoint=False,
    # Data Handlers
    mean_fvc=mean_fvc,
    std_fvc=std_fvc,
    dict_train_sequence_fvc=dict_train_sequence_fvc,
    dict_train_sequence_weekssincelastvisit=dict_train_sequence_weekssincelastvisit,
    dict_train_patients_masks_paths=dict_train_patients_masks_paths,
    dict_patients_train_ini_features=dict_patients_train_ini_features
)

model = PulmonarFibrosisEncoderDecoder(**model_inputs)

#####################################################################


#####################################################################
# 05. Web App

st.title('Dashboard - FibrosisProgression Model Predictor')
st.sidebar.subheader('1. Select a Patient')
patient = st.sidebar.selectbox('Patient: ', unique_test_patients)

# 1.Displays

st.sidebar.subheader('2. Displays')
display_scans = st.sidebar.checkbox('Display Patient Scans')
display_masks = st.sidebar.checkbox('Display Patient Masks')
display_tabular = st.sidebar.checkbox('Display Patient Tabular Data')


@st.cache
def loadCachedNumpy(path):
    return np.load(path)

if display_scans:
    patient_images_files = dict_train_patients_paths[patient]
    patient_images = loadCachedNumpy(patient_images_files[0])#st.cache(np.load)(patient_images_files[0])
    st.text(f'The patient {patient} has {patient_images.shape[0]} slices.')

    st.pyplot(plotSampleStack(patient_images, rows=4, cols=4, 
                    start_with=int(patient_images.shape[0]*0.2),
                    show_every=int((patient_images.shape[0] - int(patient_images.shape[0]*0.2)) / (4*4)),
                    figsize=(20, 20)))

if display_masks:
    patient_masks_files = dict_train_patients_masks_paths[patient]
    patient_masks = loadCachedNumpy(patient_masks_files[0])#st.cache(np.load)(patient_masks_files[0])
    st.text(f'The patient {patient} has {patient_masks.shape[0]} slices.')

    st.pyplot(plotSampleStack(patient_masks, rows=4, cols=4, 
                    start_with=int(patient_masks.shape[0]*0.2),
                    show_every=int((patient_masks.shape[0] - int(patient_masks.shape[0]*0.2)) / (4*4)),
                    figsize=(20, 20)))

if display_tabular:
    df_ini_features = st.cache(pd.DataFrame)({
        'Patient' : patient,
        'Sex' : dict_sex_inv[dict_patients_train_ini_features[patient]['Sex']],
        'Age' : unscale(dict_patients_train_ini_features[patient]['Age'], mean_age, std_age),
        'Smoking Status' : dict_smoke_inv[dict_patients_train_ini_features[patient]['SmokingStatus']],
        'Percent': unscale(dict_patients_train_ini_features[patient]['Percent'], mean_perc, std_perc),
        'Initial FVC' : unscale(dict_patients_train_ini_features[patient]['FVC'], mean_fvc, std_fvc)
    }, index=[0])
    st.dataframe(df_ini_features)

#####################################################################
 
## PreModel Viewer

st.subheader('1. Display Model Window Masks ')
st.text('Display a GIF from patients mask with selected window bounds.' )
interactive_masks = st.button('Display', key='display_1')
min_bound = st.slider('Min_Bound: ', min_value=-1_000, max_value=1_000, step=50, key='min_bound')
max_bound = st.slider('Min_Bound: ', min_value=-1_000, max_value=1_000, step=50, key='max_bound')
qt_interactive = 0
if interactive_masks:
    # -1000, 400
    buildWindowGif(dict_train_patients_masks_paths[patient], min_bound=min_bound, max_bound=max_bound, name=f'scan_gif_{qt_interactive}.gif')
    file_ = open(f"./tmp/scan_gif_{qt_interactive}.gif", 'rb')
    contents = file_.read()
    contents_url = base64.b64encode(contents).decode('utf-8')
    file_.close()
    st.markdown(f'<img src="data:image/gif;base64,{contents_url}" alt="window gif">', unsafe_allow_html=True)
    qt_interactive += 1

### Model Img viewer

st.sidebar.subheader('3. Model options ')
alpha = st.sidebar.slider('Slice Factor: ', min_value=0.5, max_value=1.0, step=0.05)
random_window = st.sidebar.checkbox('Frame Random Window')
center_crop = st.sidebar.checkbox('Center Crop')

img_size_load=(200, 200, 1)
img_size_crop=(160, 160, 1)
num_frames_batch = 32

@st.cache(suppress_st_warning=True)
def getBatch(X_generator, patient):
    patient_imgs, _ = X_generator.getOnePatient(patient)
    st.pyplot(plotSampleStack(patient_imgs[0].squeeze(), rows=4, cols=4, 
                            start_with=0,
                            show_every=2,
                            figsize=(24, 24)))

st.subheader('2. Display Patient Model Image Generator') 
get_batch = st.button('Display', key='display_2') 
if get_batch:
    X_generator = SequenceToSequenceDataGenerator(training=True, df=df_train,
                                                batch_size=1, num_frames_batch=num_frames_batch, 
                                                alpha=alpha, random_window=random_window, center_crop=center_crop,
                                                img_size_load=img_size_load, img_size_crop=img_size_crop,
                                                dict_ini_features=dict_patients_train_ini_features, 
                                                dict_patients_masks_paths=dict_train_patients_masks_paths)

    getBatch(X_generator, patient)

### Week Elapsed
st.subheader('3. Model Predictions')
st.text("Input Elapsed time(weeks) between visits. Example. '[2, 2, 4, 8, 6, 8, 12]'")
list_weeks_elapsed = st.text_area('Weeks Sequence Elapsed', height=8) 
if list_weeks_elapsed:
    try:
        list_weeks_elapsed = ast.literal_eval(list_weeks_elapsed)
    except:
        st.warning('Incorrect format of Weeks Sequence,try to type something like: [2, 4, 4, 8, 6, 8]')

initial_fvc = st.sidebar.slider('Initital FVC: ', min_value=500, max_value=6_000, step=5)
make_prediction = st.checkbox('Make prediction')
dict_predictions = {}
if make_prediction:
    # flag_prediction = True
    # model.ckpt.restore(sorted(model.ckpt_manager.checkpoints)[-1])
    X_generator = SequenceToSequenceDataGenerator(training=True, df=df_train,
                                                batch_size=1, num_frames_batch=num_frames_batch, 
                                                alpha=alpha, random_window=random_window, center_crop=center_crop,
                                                img_size_load=img_size_load, img_size_crop=img_size_crop,
                                                dict_ini_features=dict_patients_train_ini_features, 
                                                dict_patients_masks_paths=dict_train_patients_masks_paths)

    batch = X_generator.getOnePatient(patient)
    result, _, attention_plot = model.predictEvaluateModel(X_generator=None,
                                                           batch=batch,
                                                           patient=patient, 
                                                           list_weeks_elapsed=list_weeks_elapsed, 
                                                           initial_fvc=scale(initial_fvc, mean_fvc, std_fvc))
    dict_predictions['predictions'] = result
    dict_predictions['attention_plot'] = attention_plot

if 'predictions' in dict_predictions.keys():
    st.success('Prediction completed!')
    display_results = st.checkbox('Show results')
    display_attention_plot = st.checkbox('Show attention plot') 

    if display_results:
        fvc_predicted = unscale(result, mean_fvc, std_fvc).flatten().astype(int)
        st.pyplot(plotSequencePrediction(fvc_predicted,
                                         None, 
                                         list_weeks_elapsed))
        st.dataframe(pd.DataFrame({
            'Weeks Elapsed' : list_weeks_elapsed,
            'FVC - Predicted' : fvc_predicted
        }))
    if display_attention_plot:
        fvc_predicted = unscale(result, mean_fvc, std_fvc).flatten().astype(int)
        st.pyplot(plotAttention(batch[0][0].squeeze(), list_weeks_elapsed, fvc_predicted, attention_plot, alpha=0.7, max_imgs=True))
