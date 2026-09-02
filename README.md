# MeepSAT: MEEP Simulation and Analysis for Telescopes 

Modern day observations of the cosmic microwave background (CMB) require detailed understanding of various optical properties of microwave telescopes. These telescopes typically operate in the wavelength range of 1 to 10 mm while the optical systems are typically measured in meters. This usually means that 3D finite differencing time domain simulations (FDTD) simulations are prohibitively large, but lots can still be learned from 2D simulations. This repository contains a number of simulation and analysis functions that wrap around the MEEP FDTD code to help the user answer specific questions regarding the performance and potential systematics of CMB telescopes. 

The main goal of MeepSAT is to support the CMB community in the design and characterization of current and future-generation telescopes by providing a complementary modeling approach to established techniques such as physical optics, geometrical optics, and the method of moments. FDTD simulations can help probe systematic effects that are difficult to characterize using existing industry-standard software, particularly in cases involving complex geometries, lossy dielectric structures, absorber tiles, anti-reflection coatings, thermally induced deformations, detector loading, specular reflections, and optical ghosting.

## Getting Started

### ReadTheDocs Documentation

The ongoing documentation for MeepSAT can be found at: https://meepsat.readthedocs.io/en/latest/ 


### Prerequisites

- Python 3.8 or higher
- MEEP
- NumPy
- Matplotlib
- h5py

### 🛠️ Installation

Due to the complex dependencies of the MEEP FDTD software, it is highly recommended to install this package using the **Conda** package manager. More details are mentioned here in the official documentation of MEEP: https://meep.readthedocs.io/en/latest/Installation/

`Note`: We will be soon editing some segments of the MEEP code to implement effective material approximation approach in our simulations. After that, MEEPSAT will have its own version of MEEP. But for now, you can move forward with the mentioned installation guidelines.

1.  **Create a new Conda environment and Install MEEP using Conda:**
    
    **pymeep:**
    ```bash
    conda create -n meep -c conda-forge pymeep
    ```

    **If you want to install parallel version of pymeep (recommended)**
    ```bash
    conda create -n parallel_meep -c conda-forge pymeep=*=mpi_mpich_*
    ```


2.  **Activate and check if the conda environment is installed properly or not:**
    
    **pymeep:**
    ```bash
    conda activate meep
    ```

    **Install parallel pymeep and other dependencies:**
    ```bash
    conda activate parallel_meep
    ```

    **Check whether everything is working or not:**
    ```bash
    python -c 'import meep'
    ```

3.  **Install this project from the repository:**

    **For development installation (editable and **recommended** for collaborators to use this):**
    ```bash
    git clone https://github.com/aa16oaslak/MeepSAT.git
    cd MeepSAT
    pip install -e .
    ```

4. **Check if everything is working with MeepSAT import or not**
    ```bash
    python -c 'import meep as mp; import meepsat as mpsat; print("Yayy!")'
    ```

5. **For updating things to the latest version of MeepSAT, just use git pull**
    ```bash
    cd /path/to/my-private-MeepSAT-repo-head-directory
    git pull
    ```

- For Additional Checks, you can use the following piece of code to see if everything is working correctly within MeepSAT:

    ```bash
    conda activate your_meep_conda_environment
    python
    ```
    Within python environment, copy paste the following lines of code
    ```python
    import meepsat.field_analysis
    import meepsat.meep_geometry

    # Check what's available in each module
    print("Functions in meepsat.field_analysis:")
    print([name for name in dir(meepsat.field_analysis) if not name.startswith('_')])

    print("\nFunctions in meepsat.meep_geometry:")
    print([name for name in dir(meepsat.meep_geometry) if not name.startswith('_')])

    ```

    If you are getting an output with all the functions in the various modules of MeepSAT, then you are all set to do some FDTD sims!!

### Repository Structure

```text
MeepSAT/
├── meepsat/                  # Python package
│   ├── field_analysis.py     # field analysis and visualization
│   ├── meep_geometry.py      # telescope geometry components
│   ├── meshing.py            # triangular-mesh utilities
│   ├── simulator.py          # simulation initialization
│   └── stepfunctions.py      # callbacks used during simulations
├── examples/                 # notebooks, scripts, and example inputs
├── doc/meepsat_docs/         # MkDocs documentation source
├── tests/                    # lightweight unit tests
├── manuscript/               # manuscript material
├── pyproject.toml            # package metadata and dependencies
└── README.md                 # project overview and installation guide
```
