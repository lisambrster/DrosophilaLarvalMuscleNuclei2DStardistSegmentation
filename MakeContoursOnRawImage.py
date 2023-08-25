
# read in raw image, cell contours (csv files) and label image (from stardist)
# draw contours (from stardist) on raw image, and cell contours

import cv2
import numpy as np
import os
import csv
import h5py
import copy
import tifffile
import argparse

def DrawContourForLabel(label_im,im,ilabel,icolor):
    mask_indices = np.where(label_im == ilabel)
    h,w = label_im.shape
    this_mask_slice = np.zeros([h,w],dtype=np.uint8)
    this_mask_slice[mask_indices] = 255
    ret, thresh = cv2.threshold(this_mask_slice, 1, 255, 0, cv2.THRESH_BINARY)
    threshcopy = thresh.copy()
    contours, hierarchy = cv2.findContours(threshcopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print('number of contours ',len(contours)) # draw on image slice
    im = cv2.drawContours(im, contours, -1, icolor, 3)
    return im



def DrawCells(img,l3_filename,l4_filename):
    # read in associated contours
    #csv_filename1 = os.path.join(pathname, isubdir, isubdir + '_XY-VL3.csv')
    #csv_filename2 = os.path.join(pathname, isubdir, isubdir + '_XY-VL4.csv')
    csv_filename1 = os.path.join(l3_filename)
    csv_filename2 = os.path.join(l4_filename)
    sf = 0.6214809
    for icontour in range(2):
        contours = []
        contour = []
        if (icontour == 0):
            csv_filename = csv_filename1
        else:
            csv_filename = csv_filename2
        with open(csv_filename, newline='') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(myreader):
                # print(', '.join(row))
                x = row[0]
                y = row[1]
                # convert to pixels
                x = int(float(x) / sf)
                y = int(float(y) / sf)
                # print(x,y)
                contour.append([x, y])
        contours.append(contour)
        contours = np.asarray(contours)
        # draw contour on binary nuclear image
        if (icontour == 0):
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        else:
            cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    return img



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str,  default='.',
                        help='full path and name of file containing label image')
    parser.add_argument('-r', '--raw', type=str,  default='.',
                        help='full path and name of file containing raw image')

    args = parser.parse_args()
    # example label file
    #label_name = '/Users/lbrown/Documents/Mary/Lisa_Test-images_SEW/LamTest2/labels.tif'
    label_name = args.label
    # example raw image
    #raw_name = '/Users/lbrown/Documents/Mary/Lisa_Test-images_SEW/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/2022-07-08_Fz2C_L1-L2_Hoe.tif'
    raw_name = args.raw

    # example csv files
    #l3_filename = '/Users/lbrown/Documents/Mary/Lisa_Test-images_SEW/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/2022-07-08_Fz2C_L1-L2_XY-VL3.csv'
    #l4_filename = '/Users/lbrown/Documents/Mary/Lisa_Test-images_SEW/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/2022-07-08_Fz2C_L1-L2_XY-VL4.csv'
    l3_filename = os.path.join(raw_name[:-8] + '_XY-VL3.csv')
    l4_filename = os.path.join(raw_name[:-8] + '_XY-VL4.csv')

    # read raw image
    img = tifffile.imread(raw_name)
    # this is float 32-bit
    a = np.percentile(img,5)
    b = np.percentile(img,95)
    print(a,b)
    img = np.clip(img,np.percentile(img, 5),
                      np.percentile(img, 95))


    print(np.min(img),np.max(img))
    h,w = img.shape
    print(img.shape)
    rgb_img = np.zeros([h,w,3],dtype=np.uint8)

    rgb_img[:,:,0] = 255 * ((img - a)/(b-a))
    rgb_img[:,:,1] = 255 * ((img - a)/(b-a))
    rgb_img[:,:,2] = 255 * ((img - a)/(b-a))


    # draw muscle cell contours
    rgb_img = DrawCells(rgb_img,l3_filename,l4_filename)

    # draw nuclear contours
    # read label image
    lab_img = tifffile.imread(label_name)
    label_list = np.unique(lab_img)
    for ilabel in label_list:
        if (ilabel != 0):
            icolor = (0, 0, 256)  # blue
            rgb_img = DrawContourForLabel(lab_img, rgb_img, ilabel, icolor)

    # save image
    out_file_name = raw_name[:-4] + 'WithCellandNuclearContours.tif'
    tifffile.imwrite(out_file_name, rgb_img)


