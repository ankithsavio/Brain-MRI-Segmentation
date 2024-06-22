from scipy.io import loadmat
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage as ski
from skimage.filters import threshold_otsu, threshold_triangle, gaussian
from skimage.morphology import dilation, erosion
from skimage.exposure import histogram
from skimage.draw import polygon
from sklearn.cluster import KMeans
import time

def min_max_normalize(img):
    """
    Normalize an image using min-max normalization.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The normalized image.

    """
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

def z_score_normalize(image):
    """
    Normalize an image using z-score normalization.

    Parameters:
    image (numpy.ndarray): The input image to be normalized.

    Returns:
    numpy.ndarray: The normalized image.

    """
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image

from skimage import exposure, io

def histogram_normalization(image):
    """Performs histogram normalization on an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The normalized image.
    """

    normalized_image = exposure.equalize_hist(image)
    return normalized_image

def init_data(i): 
    """
    Initialize the data by loading the 'Brain.mat' file and performing min-max normalization on the T1 data.

    Parameters:
    i (int): Index of the slice to be processed.

    Returns:
    img (ndarray): Preprocessed image data.
    """
    data = loadmat('data/Brain.mat')
    T1 = data['T1']
    label = data['label']
    T1 = min_max_normalize(T1[:,:,i])
    # T1 = z_score_normalize(T1[:,:,i])
    # T1 = histogram_normalization(T1[:,:,i])
    img = T1.copy()
    return img, label

def init_3d_data():
    """
    Loads and preprocesses 3D brain data.

    Returns:
    img3d (ndarray): Preprocessed 3D brain image data.
    """
    data = loadmat('data\Brain.mat')
    T1 = data['T1']
    label = data['label']
    T1 = min_max_normalize(T1)
    # T1 = z_score_normalize(T1)
    # T1 = histogram_normalization(T1)
    img3d = T1.copy()
    return img3d

def mri_hist(img):
    """
    Display the histogram of pixel intensities in an MRI image.
    
    Parameters:
    img (numpy.ndarray): The input MRI image.
    """
    hist, bins = histogram(img, nbins=256)
    plt.xlabel('Pixel intensities')
    plt.ylabel('No of values')
    plt.bar(bins, hist)
    plt.show()

def preprocess_data(img, sigma = 1):
    """
    Preprocesses the input image by applying a Gaussian filter.
    
    Args:
        img (numpy.ndarray): The input image to be preprocessed.
        sigma (float, optional): The standard deviation of the Gaussian filter. Default is 1.
    
    Returns:
        numpy.ndarray: The preprocessed image after applying the Gaussian filter.
    """
    
    return gaussian(img, sigma = sigma)


def dilate_erode(img):
    """
    Morphological Closing. Applies dilation and erosion operations on an image until there are 2 distinct regions.

    Parameters:
    img (numpy.ndarray): The input 2D image.

    Returns:
    numpy.ndarray: The processed 2D image.

    """
    kernel = np.ones((3, 3))
    iterations = [1, 2, 3, 4, 5]
    for n in iterations:
        for i in range(n):
            img = dilation(img, kernel)
        for i in range(n):    
            img = erosion(img, kernel)
        if len(np.unique(ski.measure.label(img))) == 2:
            break
    return img

def erode_dilate(img):
    """
    Morphological Opening. Applies erosion and dilation operations on an image until there are 2 distinct regions.

    Parameters:
    img (numpy.ndarray): The input 2D image.

    Returns:
    numpy.ndarray: The processed 2D image.

    """
    kernel = np.ones((3, 3))
    iterations = [1, 2, 3, 4, 5]
    for n in iterations:
        for i in range(n):
            img = erosion(img, kernel)
        for i in range(n):
            img = dilation(img, kernel)
        if len(np.unique(ski.measure.label(img))) == 2:
            break
    return img

def dilate_erode_3d(img):
    """
    Morphological Closing. Applies dilation and erosion operations on the 3D image until there are 2 distinct regions.

    Parameters:
    img (numpy.ndarray): The input 3D image.

    Returns:
    numpy.ndarray: The processed 3D image.

    """
    kernel = ski.morphology.ball(1)
    iterations = [1, 2, 3, 4, 5]
    for n in iterations:
        for i in range(n):
            img = dilation(img, kernel)
        for i in range(n):    
            img = erosion(img, kernel)
        if len(np.unique(ski.measure.label(img))) == 2:
            break
    return img

def erode_dilate_3d(img):
    """
    Morphological Opening. Applies erosion and dilation operations on the 3D image until there are 2 distinct regions.

    Parameters:
    img (numpy.ndarray): The input 3D image.

    Returns:
    numpy.ndarray: The processed 3D image.

    """
    kernel = ski.morphology.ball(1)
    iterations = [1, 2, 3, 4, 5]
    for n in iterations:
        for i in range(n):
            img = erosion(img, kernel)
        for i in range(n):
            img = dilation(img, kernel)
        if len(np.unique(ski.measure.label(img))) == 2:
            break
    return img


def segment2d(index):
    '''
    Utilizes Contour Finding.
    Segments the 2D slice of the Brain MRI.

    This function performs segmentation on a Brain MRI image. It applies various image processing techniques
    to separate different regions of the brain, such as air, skin, skull, CSF (Cerebrospinal Fluid), GM (Gray Matter),
    and WM (White Matter).
    
    Parameters:
    index (int): Index to slice into the Brain MRI data.

    Returns:
    - preds: A numpy array representing the segmented image, where different regions are assigned different labels.
             The labels are as follows:
             - 0: Air
             - 1: Skin
             - 2: Skull
             - 3: CSF
             - 4: GM
             - 5: WM
    '''
    start_time = time.time()
    # initialize the data with min-max normalization
    img = init_data(index)
    # apply Gaussian Smoothing
    img = preprocess_data(img, 1)

    # apply triangle threshold to isolate air region
    air_thresh = threshold_triangle(img)
    air_mask = img < air_thresh
    # remove noise from the air mask
    air_mask = erode_dilate(air_mask)
    img[air_mask == 1] = 0
    img_copy = img.copy()
 

    # apply otsu threshol to get the otsu mask
    otsu_thresh = threshold_otsu(img_copy)
    otsu_mask = img_copy > otsu_thresh
    # apply contour finding algorithm
    otsu_contours = ski.measure.find_contours(otsu_mask)
    # get contours corresponding to the skin
    contour1, contour2 = otsu_contours[:2]
    
    # use the contours coordinates to create a mask    
    skin_mask = np.zeros(img_copy.shape)
    rr1, cc1 = polygon(contour1[:, 0], contour1[:, 1], skin_mask.shape)
    rr2, cc2 = polygon(contour2[:, 0], contour2[:, 1], skin_mask.shape)
    skin_mask[rr1, cc1] = 1
    skin_mask[rr2, cc2] = 0
    # close holes in the skin mask
    skin_mask = dilate_erode(skin_mask)

    # extract the brain from the otsu mask
    otsu_mask[skin_mask == 1] = 0
    # remove contiguous holes
    brain_mask = ski.morphology.remove_small_holes(otsu_mask, area_threshold= 10000)
    # remove noise from the brain mask
    brain_mask = erode_dilate(brain_mask)


    # apply air, skin and brain mask to the image
    img_copy[(skin_mask == 1) | (brain_mask == 1)] = 0
    # convert image to binary
    binary = img_copy > 0.01
    # apply contour finding algorithm
    binary_contour = ski.measure.find_contours(binary)
    contour1, contour2 = binary_contour[:2]
    noise_mask = np.zeros(binary.shape)
    rr1, cc1 = polygon(contour1[:, 0], contour1[:, 1], noise_mask.shape)
    rr2, cc2 = polygon(contour2[:, 0], contour2[:, 1], noise_mask.shape)
    noise_mask[rr1, cc1] = 1
    noise_mask[rr2, cc2] = 0

    # apply noise mask
    img_copy[noise_mask == 1] = 0
    # use clustering to strip skull mask
    n_clusters = 3
    img_reshape = img_copy.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_.reshape((img_copy.shape))
    skull_mask = labels == centroid_indices[1]

    # use air, skin, skull and noise mask to create a processed mask
    img[(skin_mask == 1) | (skull_mask == 1) | (noise_mask == 1)] = 0
    cluster_mask = img > 0.001
    # remove noise from the mask
    cluster_mask = erode_dilate(cluster_mask)


    # apply clustering on the orignal image corresponding to the processed mask
    img[cluster_mask == 0] = 0
    n_clusters = 4
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_.reshape((img.shape))
    csf_mask = labels == centroid_indices[1]
    gm_mask = labels == centroid_indices[2]
    wm_mask = labels == centroid_indices[3]

    # add each masks as labels with skin as default
    preds = np.ones(img.shape)
    preds[air_mask == 1] = 0
    preds[skull_mask == 1] = 2
    preds[csf_mask == 1] = 3
    preds[gm_mask == 1] = 4
    preds[wm_mask == 1] = 5

    end_time = time.time()
    print("Execution time:", end_time - start_time)
    return preds


def segment3d():
    '''
    Segments the whole 3D Brain MRI.

    This function performs segmentation on a 3D Brain MRI image. It applies various image processing techniques
    to separate different regions of the brain, such as air, skin, skull, CSF (Cerebrospinal Fluid), GM (Gray Matter),
    and WM (White Matter).

    Returns:
    - preds: A numpy array representing the segmented image, where different regions are assigned different labels.
             The labels are as follows:
             - 0: Air
             - 1: Skin
             - 2: Skull
             - 3: CSF
             - 4: GM
             - 5: WM
    '''
    start_time = time.time()
    # initialize the data with min-max normalization
    img = init_3d_data()

    # apply triangle threshold to isolate air region
    air_thresh = threshold_triangle(img)
    air_mask = img < air_thresh
    # remove noise from the air mask
    air_mask = erode_dilate_3d(air_mask) 
    img[air_mask == 1] = 0
    img_copy = img.copy()

    # apply otsu threshol to get the otsu mask
    otsu_thresh = threshold_otsu(img_copy)
    otsu_mask = img_copy > otsu_thresh
    # apply connected components labelling algorithm
    skin_mask = (ski.measure.label(otsu_mask) == 1)
    # close holes in the skin mask
    skin_mask = dilate_erode_3d(skin_mask)

    # apply connected components labelling algorithm
    brain_mask = (ski.measure.label(otsu_mask) == 2)
    # close holes in the brain mask
    brain_mask = dilate_erode_3d(brain_mask)
    # remove contiguous holes
    brain_mask = ski.morphology.remove_small_holes(brain_mask, area_threshold= 100000)

    # apply air, skin and brain mask to the image
    img_copy[(skin_mask == 1) | (brain_mask == 1)] = 0
    # convert image to binary
    binary = img_copy > 0.01
    # apply connected components labelling algorithm
    noise_mask = (ski.measure.label(binary) == 1)

    # apply noise mask
    img_copy[noise_mask == 1] = 0
    # use clustering to strip skull mask
    n_clusters = 3
    img_reshape = img_copy.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_.reshape((img_copy.shape))
    skull_mask = labels == centroid_indices[1]

    # use air, skin, skull and noise mask to create a processed mask
    img[(skin_mask == 1) | (skull_mask == 1) | (noise_mask == 1)] = 0
    cluster_mask = img > 0.001
    # remove noise from the mask
    cluster_mask = erode_dilate_3d(cluster_mask)

    # apply clustering on the orignal image corresponding to the processed mask
    img[cluster_mask == 0] = 0
    n_clusters = 4
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_.reshape((img.shape))
    csf_mask = labels == centroid_indices[1]
    gm_mask = labels == centroid_indices[2]
    wm_mask = labels == centroid_indices[3]

    # add each masks as labels with skin as default
    preds = np.ones(img.shape)
    preds[air_mask == 1] = 0
    preds[skull_mask == 1] = 2
    preds[csf_mask == 1] = 3
    preds[gm_mask == 1] = 4
    preds[wm_mask == 1] = 5

    end_time = time.time()
    print("Execution time:", end_time - start_time)
    return preds

def segment2d_ex(index):
    '''
    Utilizes Connected Component Labelling.
    Segments the 2D slice of the Brain MRI.

    This function performs segmentation on a Brain MRI image. It applies various image processing techniques
    to separate different regions of the brain, such as air, skin, skull, CSF (Cerebrospinal Fluid), GM (Gray Matter),
    and WM (White Matter).
    
    Parameters:
    index (int): Index to slice into the Brain MRI data.

    Returns:
    - preds: A numpy array representing the segmented image, where different regions are assigned different labels.
             The labels are as follows:
             - 0: Air
             - 1: Skin
             - 2: Skull
             - 3: CSF
             - 4: GM
             - 5: WM
    '''
    start_time = time.time()
    # initialize the data with min-max normalization
    img = init_data(index)

    # apply triangle threshold to isolate air region
    air_thresh = threshold_triangle(img)
    air_mask = img < air_thresh
    # remove noise from the air mask
    air_mask = erode_dilate(air_mask)
    img[air_mask == 1] = 0
    img_copy = img.copy()
 

    # apply otsu threshol to get the otsu mask
    otsu_thresh = threshold_otsu(img_copy)
    otsu_mask = img_copy > otsu_thresh
    # apply connected components labelling algorithm
    skin_mask = (ski.measure.label(otsu_mask) == 1)
    # close holes in the skin mask
    skin_mask = dilate_erode(skin_mask)

    # apply connected components labelling algorithm
    brain_mask = (ski.measure.label(otsu_mask) == 2)
    # close holes in the brain mask
    brain_mask = dilate_erode(brain_mask)
    # remove contiguous holes
    brain_mask = ski.morphology.remove_small_holes(otsu_mask, area_threshold= 10000)
 

    # apply air, skin and brain mask to the image
    img_copy[(skin_mask == 1) | (brain_mask == 1)] = 0
    # convert image to binary
    binary = img_copy > 0.01
    # apply connected components labelling algorithm
    noise_mask = (ski.measure.label(binary) == 1)

    # apply noise mask
    img_copy[noise_mask == 1] = 0
    # use clustering to strip skull mask
    n_clusters = 3
    img_reshape = img_copy.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_.reshape((img_copy.shape))
    skull_mask = labels == centroid_indices[1]

    # use air, skin, skull and noise mask to create a processed mask
    img[(skin_mask == 1) | (skull_mask == 1) | (noise_mask == 1)] = 0
    cluster_mask = img > 0.001
    # remove noise from the mask
    cluster_mask = erode_dilate(cluster_mask)


    # apply clustering on the orignal image corresponding to the processed mask
    img[cluster_mask == 0] = 0
    n_clusters = 4
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_.reshape((img.shape))
    csf_mask = labels == centroid_indices[1]
    gm_mask = labels == centroid_indices[2]
    wm_mask = labels == centroid_indices[3]

    # add each masks as labels with skin as default
    preds = np.ones(img.shape)
    preds[air_mask == 1] = 0
    preds[skull_mask == 1] = 2
    preds[csf_mask == 1] = 3
    preds[gm_mask == 1] = 4
    preds[wm_mask == 1] = 5

    end_time = time.time()
    print("Execution time:", end_time - start_time)
    return preds


from sklearn.metrics import jaccard_score

def calculate_jaccard_score(pred, label):
    """
    Calculate the Jaccard score between predicted and ground truth labels.

    Args:
        pred (ndarray): Predicted labels.
        label (ndarray): Ground truth labels.

    Returns:
        float: Jaccard score.

    """
    ground_truth = label.copy()  
    predicted_labels = pred.copy()  

    # Flatten the 3D masks into 1D arrays
    ground_truth_flat = ground_truth.flatten()
    predicted_labels_flat = predicted_labels.flatten()

    # Calculate the Jaccard score
    score = jaccard_score(ground_truth_flat, predicted_labels_flat, average='macro')

    print("Jaccard score:", "{:.5f}".format(score))
    return score


from sklearn.metrics import f1_score

def calculate_dice_coefficient(pred, label):
    """
    Calculate the Dice coefficient between the predicted and ground truth labels.

    Parameters:
    pred (ndarray): The predicted labels.
    label (ndarray): The ground truth labels.

    Returns:
    float: The Dice coefficient score.
    """

    ground_truth = label.flatten()
    predicted_labels = pred.flatten()
    score = f1_score(ground_truth, predicted_labels, average='macro')
    print("Dice coefficient:", "{:.5f}".format(score))
    return score

def apply_air_mask(x):
    """
    Applies an air mask to an input image.

    Parameters:
    x (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image with the air mask applied.
    numpy.ndarray: The binary mask representing the air region.

    """
    img = x.copy()
    thresh = threshold_triangle(img)
    binary = img < thresh
    binary = erode_dilate(binary)

    img[binary == 1] = 0
    
    return img, binary

def apply_skin_mask(x):
    """
    Applies a skin mask to an input image.

    Parameters:
    x (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image with the skin mask applied.
    numpy.ndarray: The binary mask representing the skin region.
    """
    img = x.copy()
    img, _ = apply_air_mask(img)
    thresh = threshold_otsu(img)
    binary = img > thresh
    binary_contour = ski.measure.find_contours(binary)
    contour1, contour2 = binary_contour[:2]
    binary_2 = np.zeros(binary.shape)
    rr1, cc1 = polygon(contour1[:, 0], contour1[:, 1], binary_2.shape)
    rr2, cc2 = polygon(contour2[:, 0], contour2[:, 1], binary_2.shape)
    binary_2[rr1, cc1] = 1
    binary_2[rr2, cc2] = 0
    binary_2 = dilate_erode(binary_2)
    img[binary_2 == 1] = 0

    return img, binary_2

def apply_brain_mask(x):
    """
    Applies a brain mask to the input image.

    Parameters:
    x (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Image with brain mask applied.
    numpy.ndarray: Binary mask of the brain region.
    """

    img = x.copy()
    img, _ = apply_air_mask(img)
    thresh = threshold_otsu(img)
    binary = img > thresh
    binary_contour = ski.measure.find_contours(binary)
    contour1, contour2 = binary_contour[:2]
    binary_2 = np.zeros(binary.shape)
    rr1, cc1 = polygon(contour1[:, 0], contour1[:, 1], binary_2.shape)
    rr2, cc2 = polygon(contour2[:, 0], contour2[:, 1], binary_2.shape)
    binary_2[rr1, cc1] = 1
    binary_2[rr2, cc2] = 0
    binary[binary_2 == 1] = 0
    binary = ski.morphology.remove_small_holes(binary, area_threshold= 10000)
    binary = erode_dilate(binary)
    img[binary == 1] = 0

    return img, binary

def apply_noise_mask(x):
    """
    Applies a noise mask to the input image.

    Parameters:
    x (ndarray): Input image.

    Returns:
    numpy.ndarray: The image with the noise mask applied.
    numpy.ndarray: The binary mask representing the noise region.

    """
    img = x.copy()
    img, _ = apply_air_mask(img)
    _, skin_mask = apply_skin_mask(img)
    _, brain_mask = apply_brain_mask(img) 
    img[(skin_mask == 1) | (brain_mask == 1)] = 0
    binary = img > 0.01
    # binary = erode_dilate(binary, 1)
    binary_contour = ski.measure.find_contours(binary)
    contour1, contour2 = binary_contour[:2]
    binary_2 = np.zeros(binary.shape)
    rr1, cc1 = polygon(contour1[:, 0], contour1[:, 1], binary_2.shape)
    rr2, cc2 = polygon(contour2[:, 0], contour2[:, 1], binary_2.shape)
    binary_2[rr1, cc1] = 1
    binary_2[rr2, cc2] = 0
    img[binary_2 == 1] = 0
    
    return img, binary_2

def apply_skull_mask(x):
    """
    Applies a skull mask to the input image.

    Parameters:
    x (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: The image with the skull mask applied.
    numpy.ndarray: The binary mask representing the skull region.
    """

    img = x.copy()
    img, noise_mask = apply_noise_mask(img)
    n=3
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_
    labels = labels.reshape((img.shape))
    binary = labels == centroid_indices[1]
    img[binary == 1] = 0
    return img, binary

def apply_csf_mask(x):
    """
    Applies a CSF (Cerebrospinal Fluid) mask to the input image.

    Parameters:
    x (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Image with CSF mask applied.
    numpy.ndarray: Binary mask indicating the CSF regions.
    """

    img = x.copy()
    img, _ = apply_air_mask(img)
    _, skin_mask = apply_skin_mask(img)
    _, skull_mask = apply_skull_mask(img)
    _, noise_mask = apply_noise_mask(img)

    img[(skin_mask == 1) | (skull_mask == 1) | (noise_mask == 1)] = 0

    binary = img > 0.001
    binary = erode_dilate(binary)
    img[binary==0] = 0

    return img, binary

def calculate_accuracy(image1, image2):
    """
    Calculates the accuracy between two images.

    Parameters:
    image1 (numpy.ndarray): The first image.
    image2 (numpy.ndarray): The second image.

    Returns:
    float: The accuracy between the two images.

    Raises:
    ValueError: If the images have different shapes.
    """

    # Convert the images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Check if the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("The images must have the same shape.")

    # Calculate the accuracy
    total_pixels = np.prod(image1.shape)
    correct_pixels = np.sum(image1 == image2)
    accuracy = correct_pixels / total_pixels

    return accuracy

def apply_kmeans(x):
    """
    Applies K-means clustering algorithm to an input image.

    Parameters:
    x (numpy.ndarray): Input image.

    Returns:
    tuple: A tuple containing binary masks for each tissue.
    """

    img = x.copy()
    img, _ = apply_csf_mask(img)
    n=4
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_
    print('shape ', kmeans.labels_.shape)
    labels = labels.reshape((img.shape))
    binary_0 = labels == centroid_indices[0]
    binary_1 = labels == centroid_indices[1]
    binary_2 = labels == centroid_indices[2]
    binary_3 = labels == centroid_indices[3]

    return binary_0, binary_1, binary_2, binary_3

def segment_it(x):
    """
    Segment the input image into different regions based on masks.

    Parameters:
    x (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Segmented image.

    """
    img = x.copy()
    air_img, air_mask = apply_air_mask(img)
    _, skull_mask = apply_skull_mask(air_img)
    _, csf_mask, gm_mask, wm_mask = apply_kmeans(air_img)

    pred = np.ones(img.shape)
    pred[air_mask == 1] = 0
    pred[skull_mask == 1] = 2
    pred[csf_mask == 1] = 3
    pred[gm_mask == 1] = 4
    pred[wm_mask == 1] = 5

    return pred

def apply_air_mask_3d(x):
    """
    Applies an air mask to the 3D input image.

    Parameters:
    x (numpy.ndarray): The input 3D image.

    Returns:
    numpy.ndarray: The image with the air mask applied.
    numpy.ndarray: The binary mask representing the air region.

    """
    img = x.copy()
    thresh = threshold_triangle(img)
    binary = img < thresh
    binary = erode_dilate_3d(binary)
    img[binary == 1] = 0
    
    return img, binary

def apply_skin_mask_3d(x):
    """
    Applies a skin mask to the 3D input image.

    Parameters:
    x (numpy.ndarray): The input 3D image.

    Returns:
    numpy.ndarray: The image with the skin mask applied.
    numpy.ndarray: The binary mask representing the skin region.
    """
    img = x.copy()
    img, _ = apply_air_mask_3d(img)
    thresh = threshold_otsu(img)
    binary = img > thresh

    binary_2 = (ski.measure.label(binary) == 1)
    binary_2 = dilate_erode_3d(binary_2)
    img[binary_2 == 1] = 0

    return img, binary_2

def apply_brain_mask_3d(x):
    """
    Applies a brain mask to the 3D input image.

    Parameters:
    x (numpy.ndarray): Input 3D image.

    Returns:
    numpy.ndarray: Image with brain mask applied.
    numpy.ndarray: Binary mask of the brain region.
    """
    img = x.copy()
    img, _ = apply_air_mask_3d(img)
    thresh = threshold_otsu(img)
    binary = img > thresh

    binary = (ski.measure.label(binary) == 2)

    binary = dilate_erode_3d(binary)
    binary = ski.morphology.remove_small_holes(binary, area_threshold= 100000)
    img[binary == 1] = 0

    return img, binary

def apply_noise_mask_3d(x):
    """
    Applies a noise mask to the 3D input image.

    Parameters:
    x (ndarray): Input 3D image.

    Returns:
    numpy.ndarray: The image with the noise mask applied.
    numpy.ndarray: The binary mask representing the noise region.

    """
    img = x.copy()
    img, air_mask = apply_air_mask_3d(img)
    _, skin_mask = apply_skin_mask_3d(img)
    _, brain_mask = apply_brain_mask_3d(img) 
    img[(skin_mask == 1) | (brain_mask == 1)] = 0
    binary = img > 0.01
   
    binary_2 = (ski.measure.label(binary) == 1)

    img[binary_2 == 1] = 0
    
    return img, binary_2

def apply_skull_mask_3d(x):
    """
    Applies a skull mask to the 3D input image.

    Parameters:
    x (numpy.ndarray): Input 3D image.

    Returns:
    numpy.ndarray: The image with the skull mask applied.
    numpy.ndarray: The binary mask representing the skull region.
    """
    img = x.copy()
    img, noise_mask = apply_noise_mask_3d(img)
    n=3
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_
    labels = labels.reshape((img.shape))
    binary = labels == centroid_indices[1]
    img[binary == 1] = 0
    return img, binary

def apply_csf_mask_3d(x):
    """
    Applies a CSF mask to the 3D input image.

    Parameters:
    x (numpy.ndarray): Input 3D image.

    Returns:
    numpy.ndarray: Image with CSF mask applied.
    numpy.ndarray: Binary mask indicating the CSF regions.
    """
    img = x.copy()
    _, skin_mask = apply_skin_mask_3d(img)
    # _, brain_mask = apply_brain_mask_3d(img) 
    _, skull_mask = apply_skull_mask_3d(img)
    _, noise_mask = apply_noise_mask_3d(img)

    img[(skin_mask == 1) | (skull_mask == 1) | (noise_mask == 1)] = 0

    binary = img > 0.001
    # binary = erode_dilate_3d(binary)
    img[binary==0] = 0

    return img, binary

def apply_kmeans_3d(x):
    """
    Applies K-means clustering algorithm to the 3D input image.

    Parameters:
    x (numpy.ndarray): Input 3D image.

    Returns:
    tuple: A tuple containing binary masks for each tissue.
    """
    img = x.copy()
    img, _ = apply_csf_mask_3d(img)
    n=4
    img_reshape = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n, random_state=42).fit(img_reshape)
    centroid_indices = np.argsort(kmeans.cluster_centers_[:, 0])
    labels = kmeans.labels_
    print('shape ', kmeans.labels_.shape)
    labels = labels.reshape((img.shape))
    binary_0 = labels == centroid_indices[0]
    binary_1 = labels == centroid_indices[1]
    binary_2 = labels == centroid_indices[2]
    binary_3 = labels == centroid_indices[3]

    return binary_0, binary_1, binary_2, binary_3

def segment_it_3d(x):
    """
    Segment the 3D input image into different regions based on masks.

    Parameters:
    x (numpy.ndarray): Input 3D image.

    Returns:
    numpy.ndarray: Segmented 3D image.

    """
    img = x.copy()
    air_img, air_mask = apply_air_mask_3d(img)
    _, skull_mask = apply_skull_mask_3d(air_img)
    _, csf_mask, gm_mask, wm_mask = apply_kmeans_3d(air_img)

    pred = np.ones(img.shape)
    pred[air_mask == 1] = 0
    pred[skull_mask == 1] = 2
    pred[csf_mask == 1] = 3
    pred[gm_mask == 1] = 4
    pred[wm_mask == 1] = 5

    return pred