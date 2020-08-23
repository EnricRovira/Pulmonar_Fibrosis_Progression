# Pulmonar_Fibrosis_Progression

## Introduction

**Fibrotic lung diseases** are an open and difficult problem nowadays, due to its variety and severity. Current methos make fibrotic lung diseases difficult to treat, even with access to a chest CT scan. Fibrotic lungs affect patients capabilty to breath correctly and in several cases this decline is fast and agressive.

This project tries to predict the severity of decline in lung function based on a CT scan of patients lungs. Breathing capacity can be measured with a spirometer, which measures the volume of air inhaled and exhaled.

The challenge is to make predictions over the time with the CT_Scans images, metadata, and baseline FVC as input. If successful, patients and their families would better understand their prognosis when they are first diagnosed with this **incurable lung disease**. Improved severity detection would also positively impact treatment trial design and accelerate the clinical development of novel treatments.

![01](./Miscellaneous/01_pulm_fib.png)

The fibrosis can be observed as scars over the lungs. This gif reflects a CT Scan with severe fibrosis

![02](./Miscellaneous/04_scan_gif.gif)


## Installation

The code is based on Jupyter notebook Framework, using Python code. The main library used is Tensorflow 2.3 but any 2.x should be supported, also i use libraries like Pandas Numpy and pydicom (in order to read dicom CT Scans) and Sciypy and OpenCV to perform Images Transformations.

Also there are few GPU commands in order to manage the memmory, in case you dont have a GPU just comment them or in case you have it you will need to install tf-nightly

`pip install --upgrade tensorflow`<br>
`pip install --upgrade tf-nightly`

In order to reproduce the results, follow these steps:

 1. Morning
 
 2. Afternoon
 
 
