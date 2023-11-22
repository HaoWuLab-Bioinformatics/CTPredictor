# CTPredictor

A comprehensive and robust framework for predicting cell types by integrating multi-omic features from single-cell Hi-C data 

## Framework
![image](Figure/framework.jpg)

(A) Data preparation. The input scHi-C dataset undergoes transformation into matrices. Our model generates sparse chromosome contact matrices for each cell from interaction pairs files in scHi-C data and simultaneously produces enhanced matrices through spatial smoothing. (B) Feature extraction. In this study, we extract small intra-domain contact probability (SICP) from the sparse matrices, smoothed small intra-domain contact probability (SSICP), and smoothed bin contact probability (SBCP) from the enhanced matrices. (C) Feature fusion. To amalgamate feature information from diverse perspectives, we introduce a fusion module. This module employs two convolutional blocks to extract more intricate and crucial features. (D) Cell classification. The fusion features are then employed to accurately predict cell types.

## Overview

The folder "4DN" contains of the preocessing of 4DN sci-HiC dataset.  
The folder "Flyamer_three" contains of the preocessing of Flyamer et al. dataset.  
The folder "Ramani" contains of the preocessing of Ramani et al. dataset.  
The folder "Lee" contains of the preocessing of Lee et al. dataset.  
The folder "Collombet" contains of the preocessing of Collombet et al. dataset.  
The folder "Nagano" contains of the preocessing of Nagano et al. dataset.  
The folder "Data_filter" contains processed data from six datasets.  

In the above six folders, each folder contains folder "generate_features" and folder "model":  
The folder "generate_features" caontains of the peocess of extracting multi-omic features.  
The folder "model" contains the framework of models.  

In folder "generate_features" of the above six folders:  
The file "generate_feature_sicp.py" is the code to extract small intra-domain contact probability (SICP) feature set.  
The file "generate_feature_ssicp.py" is the code to extract smoothed small intra-domain contact probability (SSICP) feature set.  
The file "generate_feature_sbcp.py" is the code to extract smoothed bin contact probabilit (SBCP) feature set.  

In folder "model" of the above six folders:  
The file "final_sicp_ssicp_sbcp.py" is the code of model.  
The file "focal_loss,py" is the code of loss function.  

## Dependency
Mainly used libraries:  
Python 3.6    
pytorch
xlsxwriter
sklearn  
numpy  

## Usage
First, you should extract features of data, you can run the script to extract smoothed bin contact probabilit (SBCP), small intra-domain contact probability (SICP), and smoothed small intra-domain contact probability (SSICP) features as follows:  
`python ./generate_features/generate_feature_sicp.py`  
`python ./generate_features/generate_feature_ssicp.py`  
`python ./generate_features/generate_feature_sbcp.py`    
Then run the script as follows to compile and run CTPredictor:  
`python ./model/final_sicp_ssicp_sbcp.py`     
