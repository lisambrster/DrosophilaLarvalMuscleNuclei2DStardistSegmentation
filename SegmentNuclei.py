# run 2D Stardist prediction on image ending with user specified suffix
# in the given directory


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import os

from glob import glob
import tifffile as tif
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
import ContourFunctions
import csv

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str,  default='.',
                        help='path containing one file ending with your specified suffix')
    parser.add_argument('-s', '--suffix', type=str,  default='.',
                        help='suffix of image file to be segmented such as Hoe or Lamin')
    parser.add_argument('-m', '--model', type=str,  default='.',
                        help='name of 2D stardist model')

    args = parser.parse_args()

    det_path = args.dir_path
    suffix = args.suffix
    model_name = args.model

    np.random.seed(6)
    lbl_cmap = random_label_cmap()

    subdirs = os.listdir(det_path)

    for idir in subdirs:
        #print(idir)
        #det_path = '/Users/lbrown/Documents/Mary/Bob/Files for Lisa/5_Test-images/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/'
        X = sorted(glob(det_path + '/' + idir + '/*' + suffix + '.tif'))

        if (len(X) < 1):
            print('no files found in this subdir ',idir)
            continue
        print('Running detection on ',X[0])
        nL = len(suffix)
        image_name = X[0][:(- 4 - nL)]
        img = imread(X[0])
        #X = list(map(imread, X[0]))

        if img.ndim == 2:
            n_channel = 1
        else:
            n_channel = img.shape[-1]

        axis_norm = (0,1)   # normalize channels independently

        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        model = StarDist2D(None, name=model_name, basedir='models')

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

        img = normalize(img, 1,99.8, axis=axis_norm)
        # newest lamin optimum threshold: 0.69, 0.3
        labels, details = model.predict_instances(img,prob_thresh=0.69,nms_thresh= 0.3)

        plt.figure(figsize=(8,8))
        plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
        plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
        plt.axis('off');

        # outputs: b/w binary image of nuclei
        bin_img = np.full(labels.shape,255,dtype=np.uint8)
        ind = np.where(labels > 0)
        bin_img[ind] = 0
        # remove nuclei in background (not in two muscle cells)
        print(image_name)
        save_tiff_imagej_compatible(os.path.join( image_name + 'Nuc-bin.tif'), bin_img, axes='YX')

        # csv file of contour points of nuclei and image with contours
        image_prefix = image_name[:-1]
        image_name = image_prefix + '_' + suffix + '.tif'
        contour_img, contours, cell_contours = ContourFunctions.DrawNuclearAndCellContours(image_prefix, image_name, labels)
        # csv file
        fields = ['x', 'y']

        # data rows of csv file
        nNucContours = len(contours)
        print(nNucContours)
        nuc_path = os.path.join(det_path,idir,'nuclear_contours')
        if not os.path.exists(nuc_path):
            os.makedirs(nuc_path)
        for i in range(nNucContours):
            nuc_num_str = '%02d' % i
            rows = []
            npts = len(contours[i][0])
            print('number of points ',npts)
            for ipt in range(npts):
                print(contours[i][0][ipt][0,0])
                print(contours[i][0][ipt][0,1])
                x =contours[i][0][ipt][0,0]
                y =contours[i][0][ipt][0,1]
                rows.append([str(x),str(y)])
            # name of csv file
            filename = os.path.join(nuc_path +'/nuclear_contour_' + nuc_num_str + '.csv')

            # writing to csv file
            with open(filename, 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                #csvwriter.writerow(fields)

                # writing the data rows
                csvwriter.writerows(rows)

        # contour image
        tif.imwrite(image_prefix + '_NucleiWithContours.tif', contour_img)
        # label image
        # remove nuclei outside of cells
        bg_color = 0
        labels = ContourFunctions.EraseOutsideNuclei(labels,  cell_contours,  bg_color)
        tif.imwrite(image_prefix + '_NuclearLabels.tif', labels)
        # rois for FIJI
        export_imagej_rois(image_prefix + '_rois.zip', details['coord'])