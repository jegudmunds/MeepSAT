# MeepSAT examples

The examples are intended to be followed in numerical order. The notebooks use
paths relative to this directory, so start Jupyter from `examples/`:

```bash
cd examples
jupyter notebook
```

1. [`01_simple_single_lens_ARC.ipynb`](01_simple_single_lens_ARC.ipynb) —
   simulate a 2D Gaussian beam propagating through a plano-convex lens with
   anti-reflection coatings, then compare the aperture and far-field results
   with GRASP and CST data.
2. [`02_simple_single_lens_AddedComplexities_TRV_a.ipynb`](02_simple_single_lens_AddedComplexities_TRV_a.ipynb) —
   extend the single-lens time-reverse simulation with absorbers, baffles, and
   flared structures.
3. [`03_SPIDER2_in_HPC.ipynb`](03_SPIDER2_in_HPC.ipynb) — describe the
   time-reverse SPIDER-2 workflow and results. This system is intended to be run
   on an HPC cluster using the scripts under `HPC_Tutorial/time-reverse/02_SPIDER2/`.
4. [`04_3lens_system_noARC.ipynb`](04_3lens_system_noARC.ipynb) — demonstrate
   a three-lens system without anti-reflection coatings. Its default low
   resolution is for demonstration only and should not be used for scientific
   results.

Supporting JSON configurations and comparison data are stored in
[`auxiliary_data/`](auxiliary_data/). HPC launch scripts and instructions are
stored in [`HPC_Tutorial/`](HPC_Tutorial/).
