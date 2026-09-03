# MeepSAT: MEEP Simulation and Analysis for Telescopes 

Modern day observations of the cosmic microwave background (CMB) require detailed understanding of various optical properties of microwave telescopes. These telescopes typically operate in the wavelength range of 1 to 10 mm while the optical systems are typically measured in meters. This usually means that 3D finite differencing time domain simulations (FDTD) simulations are prohibitively large, but lots can still be learned from 2D simulations. This repository contains a number of simulation and analysis functions that wrap around the MEEP FDTD code to help the user answer specific questions regarding the performance and potential systematics of CMB telescopes. 

The main goal of MeepSAT is to support the CMB community in the design and characterization of current and future-generation telescopes by providing a complementary modeling approach to established techniques such as physical optics, geometrical optics, and the method of moments. FDTD simulations can help probe systematic effects that are difficult to characterize using existing industry-standard software, particularly in cases involving complex geometries, lossy dielectric structures, absorber tiles, anti-reflection coatings, thermally induced deformations, detector loading, specular reflections, and optical ghosting.


![til](./examples/auxiliary_data/04_3lens_system_ARC/forebaffles_angle_length_TFWD_offaxis_sims_animation_Ez2_electric_field_power_anim_f150_angle_17.0_hypotenuse_800_i0.gif)


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

    print("\nFunctions in meepsat.meep_goemetry:")
    print([name for name in dir(meepsat.meep_goemetry) if not name.startswith('_')])

    ```

    If you are getting an output with all the functions in the various modules of MeepSAT, then you are all set to do some FDTD sims!!

### Repository Struture
    ```
    MeepSAT/
    ├── README.md                               # overview of the project
    ├── pyproject.toml                          # for initial Installation through pip 
    ├── data/                                   # data files used in the project
    │   ├── README.md                           # describes data from different references/softwares
    │   └── sub-directory/                      # may contain subdirectories
    ├── processed_data/                         # important files/data resulted from the analysis of the sims
    ├── manuscript/                             # manuscript of MeepSAT in latex doc
    ├── results/                                # results of the analysis (data, tables, figures)
    ├── info/                                   # contains all code related information in the project
    │   ├── LICENSE                             # license for the code
    │   ├── requirements.txt                    # code requirements and dependencies
    │   └── ...
    └── MeepSAT
        ├── bash_scripts/                       # Contains different bash scripts for future development purposes
                ├── json_to_MeepSAT_script.sh/  # This script takes a json file and directly outputs the required  python script you    need (something planned for the future)
        ├── __init__.py                         # Initialisation file
        └── field_analysis.py                   # Analysis functions for both post and during simulations
        └── simulation_2D.py                    # MeepSAT's simulator object and its corresponding functions
        └── json_to_script.py                   # Functionalities to read the json component file and output the required python script 
        └── meep_goemetry.py                    # Different components of the Telescope creating using the basic MEEP objects
        └── permittivity_components.py          # Different complex components of the telescope created for the `input_epsilon_file` parameter in MEEP.Simulation()
        └── simulator.py                        # MeepSAT Simulator intialisation
        └── stepfunctions.py                    # Different utilities (such as animation, time-averaged field extraction etc) needed during the simulation
    └── examples
        ├── project-directory/                  # Contains different example projects
                ├── sim_files/
                ├── output_files/
    └── doc/                                    # Directory for the ReadTheDocs documentation
        ├── index.rst
        └── ...
    ```
