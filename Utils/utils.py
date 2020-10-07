################################################################################################################
# Global Functions applied repeatedly in the whole project

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

def windowImage(patient_images, window_center=-750, window_width=1_000):
    patient_window_imgs = []
    for img in patient_images:
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_img = img.copy()
        window_img[window_img < img_min] = img_min
        window_img[window_img > img_max] = img_max
        patient_window_imgs.append(window_img)
    return np.asarray(patient_window_imgs)


def windowImageNorm(image, min_bound=-1_000, max_bound=400):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.
    
    return np.asarray(image)


def plotTrainHistory(history, title, scale=False):
    plt.figure(figsize=(12, 6))
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title(title)
    plt.legend()
    
    if scale:
        plt.yscale('log')
        plt.gca().grid(axis='y', which='minor')

    plt.show()
    

def plotSequencePrediction(y_pred, y_target=None, weeks_elapsed=None):
    plt.figure(figsize=(12, 6))
    y_pred = y_pred.flatten()
    if y_target is not None:
        y_target = y_target.flatten()
     
    length = np.arange(y_pred.shape[0])
    if weeks_elapsed is None:
        xticks = length
        plt.xticks(length, length)
        plt.xlabel('Prediction Sequence.')
    else:
        xticks = list(weeks_elapsed)
        plt.xticks(length, xticks)
        plt.xlabel('Weeks elapsed since last visit.')
    
    plt.plot(length, y_pred, '-b', linewidth=2, label='y_pred')
    if y_target is not None:
        plt.plot(length, y_target, '-r', linewidth=2, label='y_target')
    
    plt.legend()
    plt.title('Sequence Prediction')
    plt.ylabel('FVC Sequence (ml units)')
    plt.show()

    
def decodePatientImages(patient, dict_patients_masks_paths, image_size=(240, 240), numpy=True):
        imgs = dict_patients_masks_paths[patient]
        imgs = np.load(imgs[0])
        imgs = windowImageNorm(imgs, min_bound=-1_000, max_bound=500) #min_bound=-1_000, max_bound=400
        imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
        img_resized = tf.convert_to_tensor([tf.image.resize(tf.expand_dims(img, axis=2), image_size) for img in imgs], 
                                           dtype=tf.float32)
        
        if numpy:
            return img_resized.numpy()
        return img_resized


def customLossFunction(y_true, y_pred, std=70):
    std_clipped = tf.cast(tf.maximum(std, 70), dtype=tf.float32)
    delta = tf.cast(tf.minimum(tf.abs(y_true - y_pred), 1_000), dtype=tf.float32)
    sq2 = tf.sqrt(2.)
    loss = (delta/std_clipped) * sq2 + tf.math.log(sq2 * std_clipped)
    loss = tf.reduce_mean(loss)
    return loss


def quantileLoss(quantiles, y_true, y_pred):
    e = y_true - y_pred
    v = tf.maximum(quantiles * e, (quantiles-1) * e)
    return tf.reduce_mean(v)


def negloglik(y, p_y): 
    return -p_y.log_prob(y)


def scale(x, mean_, std_):
    return (x - mean_) / std_


def unscale(x, mean_, std_):
    return (x * std_) + mean_


def normalize(x, min_, max_):
    return (x - min_) / (max_ - min_)


def unormalize(x, min_, max_):
    return x * (max_ - min_) + min_


def getTrainValidation(df_train, df_test):
    list_test_patients = list(df_test['Patient'].unique())
    df_X_train = df_train[~df_train['Patient'].isin(list_test_patients)].reset_index(drop=True)
    df_X_val = df_train[df_train['Patient'].isin(list_test_patients)].reset_index(drop=True)
    
    return df_X_train, df_X_val

def calculateMeanImgsTrain(unique_train_patients, dict_train_patients_masks_paths):
    total_sum, total_shape = 0, 0
    for patient in tqdm(unique_train_patients, position=0):
        imgs = dict_train_patients_masks_paths[patient]
        imgs = np.load(imgs[0])
        imgs = windowImageNorm(imgs, min_bound=-1_000, max_bound=400)
        total_sum += imgs.sum()
        total_shape += imgs.flatten().shape[0]
        
    imgs_mean = total_sum / total_shape    
    return imgs_mean

def cropLung(img):
    min_, max_ = img.min(), img.max()
    if min_ == max_:
        return img
    else:
        edge_pixel_value = img[0, 0]
        mask = img != edge_pixel_value
        return img[np.ix_(mask.any(1),mask.any(0))]
    
def gaussianKernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def applyBlur(img):
    blur = gaussianKernel(3, 2, 1, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img


################################################################################################################