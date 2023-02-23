# DrosophilaLarvalMuscleNuclei2DStardistSegmentation


To install:

Assuming you have python installed,

git clone git@github.com:lisambrster/DrosophilaLarvalMuscleNuclei2DStardistSegmentation.git
pip install tensorflow\
pip install stardist


To run:

from the DrosophilaLarvalMuscleNuclei2DStardistSegmentation directory, run\
python 2D_Stardist_prediction.py -d '/Users/lbrown/Documents/Mary/Bob/Files for Lisa/5_Test-images/2022-07-08_Fz2C/2022-07-08_Fz2C_L1-L2/'

where you put your path and this path contains one file ending with 'Hoe.tif'

The outputs will be put in this path and include\
(1) an instance segmentation label image (each nucleus is labeled with a different number, and the background is 0)\
(2) roi.zip file which can be used in FIJI

