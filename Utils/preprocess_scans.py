import os
import pandas as pd
import numpy as np
import random
np.random.seed(12)

import matplotlib.pyplot as plt
import glob
import cv2

import time
from tqdm import tqdm

import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation, color
import pydicom
from sklearn.cluster import KMeans
import imageio
from joblib import parallel_backend, Parallel, delayed
import PIL


def loadSlices(patient_files):
    slices = [pydicom.read_file(s) for s in patient_files]
    slices.sort(key = lambda x: float(x.InstanceNumber))
        
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        try:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        except:
            try:
                slice_thickness = slices[0].SliceThickness
            except:
                slice_thickness = 0.5
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


def resampleImages(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = [float(scan[0].SliceThickness), 
                float(scan[0].PixelSpacing[0]), 
                float(scan[0].PixelSpacing[1])]

    spacing = np.array(spacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, spacing


def generateMarkers(image, only_internal=False):
    """
    Generates markers for a given image.
    
    Parameters: image
    
    Returns: Internal Marker, External Marker, Watershed Marker
    """
    
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    
    marker_internal = marker_internal_labels > 0
    
    if only_internal:
        return marker_internal
    
    # Creation of the External Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    
    # Creation of the Watershed Marker
    marker_watershed = np.zeros(image.shape, dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    
    return marker_internal, marker_external, marker_watershed


def seperateLungs(image, n_iters=2, only_internal=False, only_watershed=False):
    """
    Segments lungs using various techniques.
    
    Parameters: image (Scan image)
    
    Returns: 
        - Segmented Lung
        - Lung Filter
        - Outline Lung
        - Watershed Lung
        - Sobel Gradient
    """
    if only_internal:
        marker_internal = generateMarkers(image, only_internal)
    else:
        marker_internal, marker_external, marker_watershed = generateMarkers(image, only_internal)
    
    
    '''
    Creation of Sobel Gradient
    '''
    
    # Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    
    '''
    Using the watershed algorithm
    
    
    We pass the image convoluted by sobel operator and the watershed marker
    to morphology.watershed and get a matrix matrix labeled using the 
    watershed segmentation algorithm.
    '''
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    if only_watershed:
        return watershed
    
    '''
    Reducing the image to outlines after Watershed algorithm
    '''
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    
    '''
    Black Top-hat Morphology:
    
    The black top hat of an image is defined as its morphological closing
    minus the original image. This operation returns the dark spots of the
    image that are smaller than the structuring element. Note that dark 
    spots in the original image are bright spots after the black top hat.
    '''
    
    # Structuring element used for the filter
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, n_iters)
    
    # Perform Black Top-hat filter
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    '''
    Generate lung filter using internal marker and outline.
    '''
    lungfilter = np.bitwise_or(marker_internal, outline)
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    '''
    Segment lung using lungfilter and the image.
    '''
    segmented = np.where(lungfilter == 1, image, -2000*np.ones(image.shape))
    
    return segmented, lungfilter, outline, watershed, sobel_gradient


def imCropCenter(img, w, h, only_bottom=False):
    img = PIL.Image.fromarray(img)
    img_width, img_height = img.size
    left, right = (img_width - w) / 2, (img_width + w) / 2
    top, bottom = (img_height - h) / 2, (img_height + h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    if only_bottom:
        bottom -= top
        top = 0
    return np.asarray(img.crop((left, top, right, bottom)))


def saveMasks(patient, dict_paths, path, center_crop_size=80):
    try:
        patient_files = dict_paths[patient]
        patient_slices = loadSlices(patient_files)
        patient_images = getPixelsHu(patient_slices)
        imgs_after_resamp, spacing = resampleImages(patient_images, patient_slices, [1, 1, 1])
        patient_imgs = np.asarray([imCropCenter(img, 320, 320) for img in imgs_after_resamp])
        segmented_lungs_fill = np.asarray([seperateLungs(img, n_iters=2, only_internal=False, only_watershed=True) 
                                               for img in patient_imgs])
        masked_images = np.where(segmented_lungs_fill==255, patient_imgs, -2_048)
        
        file_name = patient + '_imgs_' + '.npy'
        if not os.path.exists(path + patient + '/'):
            os.mkdir(path + patient + '/')
        file_output = path + patient + '/' + file_name
        np.save(file_output, masked_images)
    except Exception as e:
        print(f'Patient {patient} failed, {e}')
        pass
    

    
def saveScans(patient, dict_paths, path, center_crop_size=80):
    try:
        patient_files = dict_paths[patient]
        patient_slices = loadSlices(patient_files)
        patient_images = getPixelsHu(patient_slices)
        imgs_after_resamp, spacing = resampleImages(patient_images, patient_slices, [1,1,1])
        patient_imgs = []
        for idx in range(imgs_after_resamp.shape[0]):
            patient_crop_img = imCropCenter(imgs_after_resamp[idx], 320, 320)
            patient_imgs.append(patient_crop_img)
            
        patient_imgs = np.asarray(patient_imgs)
        file_name = patient + '_imgs_' + '.npy'
        if not os.path.exists(path + patient + '/'):
            os.mkdir(path + patient + '/')
        file_output = path + patient + '/' + file_name
        np.save(file_output, patient_imgs)
    except:
        print(f'Patient {patient} failed')
        pass