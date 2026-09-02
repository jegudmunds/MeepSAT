This directory contains examples for running MeepSAT simulations locally or in
Slurm-based HPC environments using shell scripts.

The idea is to have a directory like this:

```
project_directory
    ├── sim_files/
        ├── project_name.py 
        ├── project_name.sh
        ├── project_name.json 
    ├── output_files/
```

Each launcher resolves paths relative to its own location, so it can be invoked
from any working directory. The default Conda environment is `parallel_meep`.
Override it without editing the script:

```bash
MEEPSAT_CONDA_ENV=my_meep_environment bash simple_single_lens_ARC.sh
```

The SPIDER2 example is a Slurm array job and should be submitted with `sbatch`.
Its MPI process count defaults to `SLURM_CPUS_PER_TASK`.
