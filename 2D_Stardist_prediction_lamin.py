# run 2D Stardist prediction on an image ending with 'Hoe.tif'
# in the given directory


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import os

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str,  default='.',
                        help='path containing one file ending Hoe.tif')


    args = parser.parse_args()

    det_path = args.dir_path
    np.random.seed(6)
    lbl_cmap = random_label_cmap()

    # read in HOE images in given directory (which will be an argument)
    #det_path = '/Users/lbrown/Documents/Mary/Bob/Files for Lisa/5_Test-images/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/'
    X = sorted(glob(det_path + '*Lam.tif'))

    nImages = len(X)
    if nImages > 1:
        print('more than Lam.tif image in directory')
        exit()
    if nImages == 0:
        print('no file ending with Lam.tif found')
        exit()

    print('Running detection on ',X[0])
    # get name of image - for output
    fulltxt = X[0]
    e = fulltxt.find('Lam')
    for i in range(e-1,0,-1):
        if (fulltxt[i] == '/'):
            s = i
            break

    image_name = fulltxt[s+1:e]

    X = list(map(imread, X))

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0,1)   # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    model = StarDist2D(None, name='stardist_lamin', basedir='models')

    # ## Prediction
    #
    # Make sure to normalize the input im
    # age beforehand or supply a `normalizer` to the prediction function.
    #
    # Calling `model.predict_instances` will
    # - predict object probabilities and star-convex polygon distances (see `model.predict` if you want those)
    # - perform non-maximum suppression (with overlap threshold `nms_thresh`) for polygons above object probability threshold `prob_thresh`.
    # - render all remaining polygon instances in a label image
    # - return the label instances image and also the details (coordinates, etc.) of all remaining polygons

    img = normalize(X[0], 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img,prob_thresh=0.8,nms_thresh= 0.3)


    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.axis('off');

    #save_tiff_imagej_compatible(os.path.join(det_path,image_name + 'image.tif'), img, axes='YX')
    save_tiff_imagej_compatible(os.path.join(det_path,image_name + 'labels.tif'), labels, axes='YX')
    export_imagej_rois(os.path.join(det_path,image_name + 'rois.zip'), details['coord'])