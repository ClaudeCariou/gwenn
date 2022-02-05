# Graph WatershEd using Nearest Neighbors (GWENN)

## Unsupervised density-based image segmentation 

GWENN is an unsupervised clustering method based on a kNN graph to propagate labels from high to low density regions of the feature space.

A multiresolution setting (Haar DWT) is also proposed to segment large-size images.

# Codes

`script_GWENN_images.py` applies the baseline GWENN method. The only parameter of the method is `K`, the number of nearest neighbors.
Notice that the number of segments/clusters need not be given, but is automatically provided at output.

`MR_gwenn.py` is the multiresolution setup of GWENN. Again, `K` must be given, as well as the number of levels `nLevel` of the DWT.

For both approaches, the set of final exemplar pixels (cluster representatives) are shown. 

# Examples

## Using `script_GWENN_images.py`
`flatboad_small.png`, 160x90 pixels, `K = 5`, `nLevel = 4`, 1066 output clusters.

## Using `MR_gwenn.py`
`colorful_houses_small.png`, 720x1280 pixels, `K = 5`, `nLevel = 4`, 1066 output clusters.

<img src="./images/original.png" width="400"> <img src="./images/segmented.png" width="400"> 


# Requirements

- Python 3.7.7
- Scikit-learn 0.23.2
- Numpy 1.19.2

# Citation

C.  Cariou  and  K.  Chehdi,  “A  new  k-nearest  neighbor  density-based  clustering 
method  and  its  application  to  hyperspectral  images,”  in  Proc.  IEEE  International 
Geoscience  and  Remote  Sensing  Symposium  (IGARSS),  Beijing,  China,  2016,  pp. 
6161–6164.

