# DrosophilaLarvalMuscleNuclei2DStardistSegmentation



Assuming you have python installed, run the following to setup this code:

```
git clone git@github.com:lisambrster/DrosophilaLarvalMuscleNuclei2DStardistSegmentation.git
pip install tensorflow
pip install stardist
```

# 9/2023 Instructions

` python SegmentNuclei.py -d ~/Documents/Mary/VictoriaMuscle/testdir -s Hoe -m stardist_hoe`

* -d is the top-level directory where each subdirectory has the cell csv files and the image file with the specified suffix
* -s is suffix of the image file - for example, here the image file will end with Hoe.tif
* -m is the name of the model, all the models are the model subdirectory of this repository

This code assumes the csv files for the cell contours are in each sub-directory.

The outputs are in the same sub-directory as the input and include:

* \*_NuclearLabels.tif is a label or mask image where each nucleus has a different label number, only nuclei inside the cells are included
* \*_Nuc_bin.tif is a black and white mask image of all the nuclei found by the Stardist model, includes all nuclei inside or outside the cells
* \*_rois.zip are the contours of the nuclei which can be read in and drawn by FIJI
* \*_NucleiWithContours.tif shows the original image with the contours of the muscle cells and nuclei
* nuclear_contours is a sub-directory which has a csv file for each nuclear contour
  

# Original Instructions

From the DrosophilaLarvalMuscleNuclei2DStardistSegmentation directory, run

`python 2D_Stardist_prediction.py -d '/Users/lbrown/Test-images/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/'`

where you put your path and this path contains one file ending with 'Hoe.tif'

The outputs will be put in the same path and include:
* labels.tif -- an instance segmentation label image (each nucleus is labeled with a different number, and the background is 0)
* roi.zip file which can be used in FIJI

To draw contours of labels on image and the contours of the two muscle cells (from the csv files) run:

`python MakeContoursOnRawImage.py -l label_image -r raw_image`

This assumes cell csv files and the label image (labels.tif) are in the same directory as the raw image

<img src='https://github.com/lisambrster/DrosophilaLarvalMuscleNuclei2DStardistSegmentation/blob/main/viz.png'>


