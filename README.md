# NGC6153 MUSE ANALYSIS 

This is a set of python scripts developed to analyse MUSE data of the planetary nebula NGC 6153. The data have been published in Gómez-Llanos et al. (2024)

# Recomended installation

HIGHLY RECOMMENDED: Install conda and use the following environment.

1) Verify that you have Conda installed.

```
conda --version
```

If you see the CONDA version, it means it's installed. In this case, skip to step 4).

2) Download CONDA (latest version as of 03/11/2023).

For Mac users:

```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-MacOSX-x86_64.sh
```

For Linux users:

```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```

3) Install conda:

For Mac users:

```
bash Anaconda3-2023.09-0-MacOSX-x86_64.sh
```

For Linux users:

```
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

4) Once CONDA is installed, create an environment named e.g. 'MUSE_PN'

```
conda create -n MUSE_PN "python==3.10.13" numpy matplotlib pandas scipy astropy h5py joblib catboost ipykernel SQLAlchemy pymysql jupyterlab tensorflow
```

5) Activate the newly created environment:

```
conda activate MUSE_PN
```

6) Install PyNeb and PyCloudy:
```
pip install -U PyNeb
```
```
pip install -U PyCloudy
```

7) Install ai4neb
```
pip install -U git+https://github.com/VGomezLlanos/AI4neb.git
```

8) Verifiy the  configuration of paths and data files. This should be done in the constants/observation_parameters.py file.

In the observations_parameters.py file in the utils folder you need to set up the working directories:

- DATA_DIR is the path to the renamed fits files of the extracted line emission maps
- OBJ_NAME is the name of your object
- OBS_NAME is the format of the names of the fits files
- OBS_INT_FILE is the path to the observed fluxes of the integrated spectrum. They are in a pyneb format especified by FILE_FORMAT_INT
- FIRS_DIR is the path to the gaussian fit .fits files of the extracted emission line maps
   
Other parameters of interest can be modified in such file

9) The figures and tables of the paper can be created running the code figuras_asrticulo.py

10) ICFs and elemental abundances using machine learning techniques are computed using the notebook ICF_ABUND_NGC6153.ipynb

# Fable-Optimization
