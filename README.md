![finalpipelinefigure](https://github.com/user-attachments/assets/4457b9e0-1a51-467b-9a29-484bfe0b304e)

# Introduction
The exposome refers to the totality of environmental, behavioral, and lifestyle exposures an individual experiences throughout one’s lifetime. Due to the modifiability of exposures, identifying the risk exposures on a disease is crucial for effective intervention and prevention of the disease. However, traditional analytical methods struggle to capture the complexities of exposome data: nonlinear effects, correlated exposures, and potential interplay with genetic effects. To address these challenges and accurately estimate exposure effects on complex diseases, we developed DeepEXPOKE, a deep learning framework integrating two types of knockoff features: statistical knockoffs (statKO) and polygenic risk score as knockoffs (PRSKO). DeepEXPOKE-statKO controls exposure correlation and DeepEXPOKE-PRSKO isolates genetic effects, while both can capture nonlinear effects. DeepEXPOKE is meant for UK BioBank data as of now, and will be adapted for additional data types and PRS scoring methods accordingly. The example .ipynb's do not need genetic data or PRS scoring scripts ran, but for any other data, users are asssumed to have their unique UKB feature data and chromosome .bed/.bim/.fam files downloaded to their cluster/local machines.
####

# Simulated example data
The example data files in this repository are fully simulated and do not contain UK Biobank participant-level values. The original data files were used only to determine the number of rows and to preserve the original file schemas, including column names. No original participant values, row-level records, identifiers, or sampled rows were retained in the simulated files.

The simulated files were generated with Python using `numpy` and `pandas`, with a fixed random seed for reproducibility:

```python
np.random.default_rng(20260511)
```

All five simulated files contain 82,894 rows, matching the row count of the original example files. Column names were preserved exactly.

Simulated files:
- `data/simulateddata_42p_82894samples_02-23-24_incidence_X.csv`
- `data/simulateddata_42p_82894samples_02-23-24_incidence_Y.csv`
- `data/simulateddata_42p_82894samples_02-23-24_incidence_XY.txt`
- `data/simulateddata_02-22-24-incidencefeatures_X.csv`
- `data/simulateddata_02-22-24-incidencefeatures_Y.csv`

The two X files contain the same simulated 42-feature matrix. One file uses generic feature names `F1` through `F42`; the other preserves the descriptive feature names from the original incidence features file. The two Y files contain the same simulated binary outcome vector, with the original headers preserved as `Y` and `Sepsis`. The XY `.txt` file is a tab-delimited concatenation of the simulated X matrix and simulated Y vector.

Continuous features were generated independently from preset synthetic distributions, then clipped to preset ranges and rounded. These parameters were selected to produce plausible example values for software demonstration, not to reproduce UK Biobank marginal distributions or correlations. Binary features were generated as Bernoulli random variables.

Feature generation parameters:

| Feature | Synthetic generation method |
| --- | --- |
| F1 | Normal mean 1.45, SD 0.25, clipped 0.60-2.50, rounded to 4 decimals |
| F2 | Normal mean 1.05, SD 0.22, clipped 0.35-2.20, rounded to 4 decimals |
| F3 | Normal mean 27.5, SD 5.2, clipped 16.0-52.0, rounded to 4 decimals |
| F4 | Normal mean 2.38, SD 0.12, clipped 1.85-3.05, rounded to 3 decimals |
| F5 | Normal mean 78.0, SD 18.0, clipped 35.0-180.0, rounded to 1 decimal |
| F6 | Normal mean 36.0, SD 8.0, clipped 20.0-90.0, rounded to 1 decimal |
| F7 | Normal mean 1.45, SD 0.35, clipped 0.45-3.20, rounded to 4 decimals |
| F8 | Normal mean 169.0, SD 9.0, clipped 145.0-200.0, rounded to 1 decimal |
| F9 | Normal mean 16.0, SD 3.2, clipped 6.0-32.0, rounded to 2 decimals |
| F10 | Normal mean 16.1, SD 3.2, clipped 6.0-32.0, rounded to 2 decimals |
| F11 | Normal mean 5.2, SD 1.1, clipped 2.4-9.5, rounded to 4 decimals |
| F12 | Normal mean 79.0, SD 11.0, clipped 45.0-125.0, rounded to integers |
| F13 | Normal mean 132.0, SD 18.0, clipped 85.0-220.0, rounded to integers |
| F14 | 65% set to 0; otherwise random integer from 25 to 80 |
| F15 | Categorical value from {0, 1, 2, 3} with probabilities 0.88, 0.09, 0.025, 0.005 |
| F16 | Normal mean 24.0, SD 12.0, clipped 5.0-120.0, rounded to 2 decimals |
| F17 | Normal mean 43.0, SD 3.0, clipped 30.0-55.0, rounded to 2 decimals |
| F18 | Normal mean 82.0, SD 24.0, clipped 30.0-220.0, rounded to 1 decimal |
| F19 | Normal mean 25.0, SD 9.0, clipped 8.0-120.0, rounded to 1 decimal |
| F20 | Log-normal median 2.2, sigma 0.8, clipped 0.1-30.0, rounded to 2 decimals |
| F21 | Normal mean 2.2, SD 1.0, clipped 0.2-8.0, rounded to 2 decimals |
| F22 | Normal mean 5.2, SD 1.1, clipped 3.0-14.0, rounded to 3 decimals |
| F23 | Normal mean 20.0, SD 6.0, clipped 5.0-45.0, rounded to 3 decimals |
| F24 | Normal mean 3.1, SD 0.9, clipped 0.7-7.0, rounded to 4 decimals |
| F25 | Log-normal median 45.0, sigma 1.0, clipped 2.0-280.0, rounded to 2 decimals |
| F26 | Log-normal median 22.0, sigma 1.1, clipped 1.0-450.0, rounded to 1 decimal |
| F27 | Normal mean 1.15, SD 0.18, clipped 0.55-2.00, rounded to 3 decimals |
| F28 | Normal mean 72.0, SD 25.0, clipped 10.0-180.0, rounded to 1 decimal |
| F29 | Normal mean 45.0, SD 22.0, clipped 5.0-160.0, rounded to 2 decimals |
| F30 | Normal mean 70.0, SD 28.0, clipped 10.0-180.0, rounded to 1 decimal |
| F31 | Log-normal median 4.0, sigma 1.0, clipped 0.1-30.0, rounded to 3 decimals |
| F32 | Normal mean 9.0, SD 4.5, clipped 2.0-40.0, rounded to 2 decimals |
| F33 | Normal mean 71.0, SD 4.0, clipped 55.0-88.0, rounded to 2 decimals |
| F34 | Log-normal median 1.7, sigma 0.6, clipped 0.3-8.0, rounded to 3 decimals |
| F35 | Normal mean 310.0, SD 75.0, clipped 120.0-650.0, rounded to 1 decimal |
| F36 | Normal mean 5.2, SD 1.5, clipped 2.0-14.0, rounded to 2 decimals |
| F37 | Normal mean 48.0, SD 22.0, clipped 5.0-140.0, rounded to 1 decimal |
| F38 | Bernoulli probability 0.11 |
| F39 | Bernoulli probability 0.06 |
| F40 | Bernoulli probability 0.08 |
| F41 | Bernoulli probability 0.26 |
| F42 | Bernoulli probability 0.13 |

The simulated binary outcome was generated from the simulated feature matrix using a synthetic logistic model:

```python
logit = (
    -2.45
    + 0.025 * (F3 - 27.5)
    + 0.006 * (F13 - 132.0)
    + 0.45 * F38
    + 0.35 * F39
    + 0.40 * F40
    + 0.25 * F41
    + 0.20 * F42
)
p = 1 / (1 + exp(-logit))
Y = Bernoulli(p)
```

After the simulated files were written, the original five data files were removed from the repository copy. The notebooks were updated to reference the `simulateddata_...` filenames and local `data/` paths.

# 1) Install and test the DAG-DeepVASE package
1. Install and test Dr. Zhenjiang Fan's DAG-DeepVASE algorithm [here](https://github.com/ZhenjiangFan/DAG-deepVASE). 
2. Follow the instructions and set up DAG-DeepVASE on your local or remote computing platform. Some of the DAG-DeepVASE framework will be used as part of DeepEXPOKE.

# Important DAG-DeepVASE Package Requirements

:stars: rpy2\
:stars: keras (:bell: please use the version of 2.3.1, which can be installed using "pip install Keras==2.3.1".)\
:stars: Deeplift (:bell: please use the version of 0.5.1-theano, which should be downloaded online.)\
:stars: Tensorflow (:bell: please use the version of 2.2.0, which can be installed using "pip install tensorflow==2.2.0".)\
:stars: jpype\
:stars: R environment\
:stars: Java environment

# 2) Install NEXTFLOW and pgsc_calc (Not necessary for running DeepEXPOKE example code)
1. Install and test Nextflow and pgsc_calc [here](https://pgsc-calc.readthedocs.io/en/latest/). 
2. Follow the instructions and set up pgsc_calc for use on your local or remote computing platform.
3. Create your test samplesheet as per ``pgsc_calc`` documentation.
4. Run pgsc_calc via nextflow with this command (replacing the PRS with your desired trait IDs/score IDs):
``./nextflow run pgscatalog/pgsc_calc \
 -profile conda \
 --input ukb_prs_samplesheet_final.csv --target_build GRCh37 \
 --pgs_id PGSXXXXXX, PGSXXXXXX, (insert the actual IDs here instead of XXXXXX) \
 --min_overlap 0.75 \
 --run_ancestry ./pgsc_calc.tar.zst``


# 3) Run the following example .ipynb notebooks using Jupyter Lab/your preferred python IDE on the example data provided in the DeepEXPOKE repository in the following order:
1. ``Dataset_preprocessing.ipynb``
2. ``W_stats_all42-DeepEXPOKE.ipynb``
3. ``imbalanced+RF-DeepEXPOKE.ipynb``
4. Optional: Run ``final_LR+RF+XGB_featureselection.ipynb`` for feature selection comparisons.

The example PRS knockoffs file is in a .7zip format; please unzip it before using in the example notebook. 
Please refer to the ``featuremapping.csv`` file for interpretation of example causal results. 

 

# Acknowledgements and References

#### :trophy: Several components of the DeepEXPOKE codebase come from the following projects:
:star: The MGM Java implemention in DAG-DeepVASE is from [causalMGM](https://github.com/benoslab/causalMGM) and [TetradLite](https://github.com/benoslab/tetradLite).\
:star: The DeepPINK implementation in DAG-DeepVASE is from [DeepPINK](https://github.com/younglululu/DeepPINK).\
:star: The FDR filter function in DAG-DeepVASE is from [DeepKnockoffs](https://github.com/msesia/deepknockoffs).\
:star: The Python implementation of Degenerate Gaussian (DG) algorithm in DeepEXPOKE is based on its Java version from [Tetrad](https://www.ccd.pitt.edu).\
:star: The implementation of the PC algorithm used in DeepEXPOKE is from [pcalg](https://github.com/keiichishima/pcalg).\
:star: The polygenic risk scoring algorithm used in DeepEXPOKE is from [pgsc_calc](https://pgsc-calc.readthedocs.io/en/latest/).


# Contact
:email: If you have any questions, bug reports, or feedback, please contact either HJ Park or Aditya Sriram via either of the following emails:
<p align="center">
    :e-mail: hyp15@pitt.edu
    :e-mail: ads303@pitt.edu
</p>
    
