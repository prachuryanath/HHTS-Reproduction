# HHTS-Reproduction
Reproducibility Report on Hierarchical Histogram Threshold SegmentationAuto-terminating High-detail Oversegmentation
# Experimental Setup
* Hardware Environment
* Processor: Ryzen 5
* RAM: 16 GB
* Graphics: Nvidia GTX 1650 Ti
# Software Environment
* Operating System: Windows 11
* Dataset Preparation and Preprocessing
* The project uses the BSDS300 dataset (Martin et al., 2001), which includes 300 images with both color and grayscale segmentations.
* The dataset is divided into:
Training set: 200 images
Test set: 100 images
* Choice of Hyperparameters :
| Parameter  | Value |
| ------------- | ------------- |
| Superpixels  | 500  |
| Minimum detail size  | 64  |
| Split threshold  | 0  |
| Histogram bins  | 32  |

# Environment Setup
## Installation
* Clone the repository: git clone https://github.com/prachuryanath/HHTS-Reproduction.git
* cd HHTS-Reproduction
* Create environment : python -m venv .venv
* Activate environment : source .venv/bin/activate (for Linux), .venv/Source/activate (for Windows)
* Install dependencies using pip: pip install -r requirements.txt
## Libraries Used
* NumPy
* OpenCV-Python
* Typing
* Bisect
