
# Introduction
The exposome refers to the totality of environmental, behavioral, and lifestyle exposures an individual experiences throughout one’s lifetime. Due to the modifiability of exposures, identifying the risk exposures on a disease is crucial for effective intervention and prevention of the disease. However, traditional analytical methods struggle to capture the complexities of exposome data: nonlinear effects, correlated exposures, and potential interplay with genetic effects. To address these challenges and accurately estimate exposure effects on complex diseases, we developed DeepEXPOKE, a deep learning framework integrating two types of knockoff features: statistical knockoffs (statKO) and polygenic risk score as knockoffs (PRSKO). DeepEXPOKE-statKO controls exposure correlation and DeepEXPOKE-PRSKO isolates genetic effects, while both can capture nonlinear effects.
####

# Install and test the DAG-DeepVASE package
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

# Install NEXTFLOW and pgsc_calc
1. Install and test Nextflow and pgsc_calc [here](https://pgsc-calc.readthedocs.io/en/latest/). 
2. Follow the instructions and set up pgsc_calc for use on your local or remote computing platform.

# Run the following .ipynb notebooks on the example data in the following order:
1. Install and test Nextflow and pgsc_calc [here](https://pgsc-calc.readthedocs.io/en/latest/). 
2. Follow the instructions and set up pgsc_calc for use on your local or remote computing platform.

 

# Acknowledgement and References

#### :trophy: Some components of this project come from the follwing projects:
:star: The MGM Java implemention is from [causalMGM](https://github.com/benoslab/causalMGM) and [TetradLite](https://github.com/benoslab/tetradLite).\
:star: The DeepPINK implementation is from [DeepPINK](https://github.com/younglululu/DeepPINK).\
:star: The FDR filter function is from [DeepKnockoffs](https://github.com/msesia/deepknockoffs).\
:star: The Python implementation of DG algorithm is based on its Java version from [Tetrad](https://www.ccd.pitt.edu).\
:star: The implementation of the PC algorithm used in this project is from [pcalg](https://github.com/keiichishima/pcalg).

# Contact
:email: Please let us know if you have any questions, bug reports, or feedback via the following email:
<p align="center">
    :e-mail: hyp15@pitt.edu
</p>
    

