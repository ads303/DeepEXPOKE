
# Introduction
The exposome refers to the totality of environmental, behavioral, and lifestyle exposures an individual experiences throughout one’s lifetime. Due to the modifiability of exposures, identifying the risk exposures on a disease is crucial for effective intervention and prevention of the disease. However, traditional analytical methods struggle to capture the complexities of exposome data: nonlinear effects, correlated exposures, and potential interplay with genetic effects. To address these challenges and accurately estimate exposure effects on complex diseases, we developed DeepEXPOKE, a deep learning framework integrating two types of knockoff features: statistical knockoffs (statKO) and polygenic risk score as knockoffs (PRSKO). DeepEXPOKE-statKO controls exposure correlation and DeepEXPOKE-PRSKO isolates genetic effects, while both can capture nonlinear effects. DeepEXPOKE is meant for UK BioBank data as of now, and will be adapted for additional data types and PRS scoring methods accordingly. Users are asssumed to already have their feature data and chromosome .bed/.bim/.fam files already downloaded to their cluster/local machines.
####

# 1) Install and test the DAG-DeepVASE package
1. Install and test DAG-DeepVASE [here](https://github.com/ZhenjiangFan/DAG-deepVASE). 
2. Follow the instructions and set up DAG-DeepVASE on your local or remote computing platform. Some of the DAG-DeepVASE framework will be used as part of DeepEXPOKE.

# Important DAG-DeepVASE Package Requirements

:stars: rpy2\
:stars: keras (:bell: please use the version of 2.3.1, which can be installed using "pip install Keras==2.3.1".)\
:stars: Deeplift (:bell: please use the version of 0.5.1-theano, which should be downloaded online.)\
:stars: Tensorflow (:bell: please use the version of 2.2.0, which can be installed using "pip install tensorflow==2.2.0".)\
:stars: jpype\
:stars: R environment\
:stars: Java environment

# 2) Install NEXTFLOW and pgsc_calc
1. Install and test Nextflow and pgsc_calc [here](https://pgsc-calc.readthedocs.io/en/latest/). 
2. Follow the instructions and set up pgsc_calc for use on your local or remote computing platform.
3. Create your test samplesheet as per ``pgsc_calc`` documentation.
4. Run pgsc_calc via nextflow with this command (replacing the PRS with your desired trait IDs/score IDs):
``./nextflow run pgscatalog/pgsc_calc \
 -profile conda \
 --input ukb_prs_samplesheet_final.csv --target_build GRCh37 \
 --pgs_id PGS000671, (insert other IDs here) \
 --min_overlap 0.75 \
 --run_ancestry ./pgsc_calc.tar.zst``


# 3) Run the following .ipynb notebooks using Jupyter Lab/your preferred python IDE on the example data provided in the DeepEXPOKE repository in the following order:
1. ``Dataset_preprocessing.ipynb``
2. ``W_stats_all42-DeepEXPOKE.ipynb``
3. ``imbalanced+RF-DeepEXPOKE.ipynb``
4. Optional: Run ``final_LR+RF+XGB_featureselection.ipynb`` for feature selection comparisons.

The example PRS knockoffs file is in a .7zip format; please unzip it before using in the example notebook. 

 

# Acknowledgements and References

#### :trophy: Some components of this project come from the follwing projects:
:star: The MGM Java implemention is from [causalMGM](https://github.com/benoslab/causalMGM) and [TetradLite](https://github.com/benoslab/tetradLite).\
:star: The DeepPINK implementation is from [DeepPINK](https://github.com/younglululu/DeepPINK).\
:star: The FDR filter function is from [DeepKnockoffs](https://github.com/msesia/deepknockoffs).\
:star: The Python implementation of DG algorithm is based on its Java version from [Tetrad](https://www.ccd.pitt.edu).\
:star: The implementation of the PC algorithm used in this project is from [pcalg](https://github.com/keiichishima/pcalg).\
:star: The polygenic risk scoring algorithm used in this project is from [pgsc_calc](https://pgsc-calc.readthedocs.io/en/latest/).


# Contact
:email: Please let us know if you have any questions, bug reports, or feedback via the following email:
<p align="center">
    :e-mail: hyp15@pitt.edu
    :e-mail: ads303@pitt.edu
</p>
    

