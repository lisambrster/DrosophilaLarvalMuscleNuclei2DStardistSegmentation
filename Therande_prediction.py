# run 2D Stardist prediction on an image ending with 'Hoe.tif'
# in the given directory


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
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
    parser.add_argument('-i', '--image_path_and_name', type=str,  default='./MAX_D7_KO2_Hoe_60X_001.tif',
                        help='path and name of image (assumes tif image)')


    args = parser.parse_args()

    image_name = args.image_path_and_name

    np.random.seed(6)
    lbl_cmap = random_label_cmap()

    print('Running detection on ',image_name)
    # get name of image - for output
    output_name = image_name[:-4] + '_labels.tif'

    X = imread(image_name)

    n_channel = 1 if X.ndim == 2 else X.shape[-1]
    axis_norm = (0,1)   # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    model = StarDist2D(None, name='stardistTherande', basedir='models')

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

    img = normalize(X, 1,99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)

    plt.figure(figsize=(8,8))
    plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
    plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    plt.axis('off');

    save_tiff_imagej_compatible(output_name, labels, axes='YX')
    export_imagej_rois(os.path.join(output_name + 'rois.zip'), details['coord'])