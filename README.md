# Pulmonar_Fibrosis_Progression

## Introduction

**Fibrotic lung diseases** are an open and difficult problem nowadays, due to its variety and severity. Current methos make fibrotic lung diseases difficult to treat, even with access to a chest CT scan. Fibrotic lungs affect patients capabilty to breath correctly and in several cases this decline is fast and agressive.

This project tries to predict the severity of decline in lung function based on a CT scan of patients lungs. Breathing capacity can be measured with a spirometer, which measures the volume of air inhaled and exhaled.

The challenge is to make predictions over the time with the CT_Scans images, metadata, and baseline FVC as input. If successful, patients and their families would better understand their prognosis when they are first diagnosed with this **incurable lung disease**. Improved severity detection would also positively impact treatment trial design and accelerate the clinical development of novel treatments.

![01](./Miscellaneous/01_pulm_fib.png)

The fibrosis can be observed as scars over the lungs. This gif reflects a CT Scan with severe fibrosis

![02](./Miscellaneous/04_scan_gif.gif)


## Installation

The code is based on Jupyter notebook Framework, using Python code. The main library used is Tensorflow 2.3 but any 2.x should be supported, as well you will need libraries like Pandas, Numpy, Pydicom (in order to read dicom CT Scans), Sciypy, OpenCV to perform Images Transformations.

Also there are few GPU commands in order to manage the memmory, in case you dont have a GPU just comment them or run it on [Google Colab](https://colab.research.google.com/). and in case you want to run it on your local machine with GPU you will need to install tf-nightly.

`pip install --upgrade tensorflow`<br>
`pip install --upgrade tf-nightly`

In order to reproduce the results, follow these steps:

 1. **[Pending]**
 
 2. **[Pending]**
 
 3. **[Pending]**
 
 
 ## Data
 
1. The Data Zip File size is about 20GB, so be sure you have enough space before unzipping.
2. The Dataset consists on 2 folders (train/ & test/), in which you will find a unique folder for each patient CT-Scan, The test patients are a subset of train patients. Besides there are a train.csv and test.csv files containing the patients metadata.

If you want acces to the data you will have to mail <enric.rovira96@gmail.com> and attach an axplanation about how are going to use the data.

Extact the contents of the zip file into your local machine or [Google Colab](https://colab.research.google.com/)



 ## Model
 
 ## Results & Metrics
 
 ## Conclusions & Further Steps




