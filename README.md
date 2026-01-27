# DEMO-EMReF: Enhancing interpretability of cryo-EM maps with hybrid attention transformers [![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)](https://pytorch.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# Quick Start with Linux
If you use the precompiled binary `DEMO-EMReF`, no Python environment is required. When you run the binary, it may appear to be unresponsive for a short period. This is normal, as the program is loading cached data and initializing the model. Please be patient; processing will start automatically once the cache is loaded.

# Show help
```bash
./dist/DEMO-EMReF -h
```
# Basic usage
```bash
./dist/DEMO-EMReF -F /path/to/input_map.mrc -o /path/to/output_map_refined.mrc
```
# Full example
```bash
./dist/DEMO-EMReF -F /path/to/input_map.mrc -o /path/to/output_map_refined.mrc --mode HR --gpu 0 -b 8 -s 24
```

If you want to download the full example, please use the following link: [Example Download](https://github.com/zhouxglab/DEMO-EMReF/releases/download/v1.0/example.zip)

After downloading, unzip the file to your working directory:

```bash
unzip example.zip -d ./example
```

# Options in DEMO-EMReF
| Option               | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `-F`                 | Input cryo-EM / cryo-ET map (MRC\MAP format)                                 |
| `-o`                 | Output refined map                                                           |
| `--mode {HR,MR,ET}`  | Inference mode (default: HR)                                                 |
| `--config`           | Custom JSON config file                                                      |
| `--gpu`              | GPU ID(s), e.g. `0` or `0,1`                                                 |
| `-b, --batch_size`   | Batch size (default: 8, increase if GPU memory allows)                       |
| `-s, --stride`       | Sliding window stride (default: 24, smaller gives better quality but slower) |
| `-m, --mask_map`     | Mask map                                                                     |
| `-p, --mask_str`     | Structure-based mask (PDB/CIF)                                               |
| `-c, --mask_contour` | Mask contour level                                                           |
| `--inverse_mask`     | Invert mask region                                                           |
| `--interp_back`      | Interpolate output back to original voxel size                               |
# Local Installation (Conda Environment):

Clone the Repository
```bash
git clone https://github.com/zhouxglab/DEMO-EMReF.git
cd DEMO-EMReF
```

Quick installation using the provided YAML file:
```bash
conda env create -f DEMO-EMReF_env.yml
conda activate DEMO-EMReF_env
```
This command will create a Python conda virtual environment named "DEMO-EMReF_env" and install all the required packages.

# Python package requirements:
    pytorch (2.0.1) (https://pytorch.org)
    pytorch-cuda (11.8) (https://pytorch.org)
    biopython (1.73) (https://biopython.org/)
    numpy (1.24.4) (https://www.numpy.org)
    einops (0.8.0) (https://einops.rocks/)
    mrcfile (1.5.3) (https://github.com/ccpem/mrcfile)
    timm(1.0.11) (https://github.com/rwightman/pytorch-image-models)
    tqdm (4.65.0) (https://github.com/tqdm/tqdm)
    
# Full example
```bash
python predict/predict.py -F /path/to/input_map.mrc -o /path/to/output_map_refined.mrc --mode HR --gpu 0 -b 8 -s 24
```

# Compile Accelerated Interpolation Module (interp3d)
For faster processing (especially when using --interp_back), we recommend compiling the Fortran-based interp3d module using f2py.
# Step 1: Activate environment
```bash
conda activate DEMO-EMReF_env
```
# Step 2: Compile
```bash
f2py -c ./interp3d.f90 -m interp3d
```
`This generates a .so file (e.g., interp3d.cpython-*.so). Keep this file in the root directory with the .py scripts.`

`Prerequisites: Requires a Fortran compiler (e.g., gfortran).`

`Ubuntu/Debian: sudo apt-get install gfortran`

`Conda: conda install -c conda-forge gfortran==11.4`
# Step 3: Custom Compiler (If needed) If gfortran is not in your PATH, specify it manually:
```bash
f2py -c ./interp3d.f90 -m interp3d --fcompiler=gnu95 --f77exec=/path/to/gfortran --f90exec=/path/to/gfortran
```
For more information about f2py, please refer to the [official documentation](https://numpy.org/doc/stable/f2py/).

How it works: The interp3d module provides high-performance 3D cubic interpolation. The main script (predict.py) automatically detects if the compiled module exists and uses it for acceleration.

# How to Run DEMO-EMReF:
```bash
python predict.py -F /path/to/input_map.mrc -o /path/to/output_map_refined.mrc [Options]
```
# Notes：
The executable version bundles all dependencies and does not require Conda or Python.
The source version requires the Conda environment described above.

# Contact
Please report bugs and questions to zxg@zjut.edu.cn
