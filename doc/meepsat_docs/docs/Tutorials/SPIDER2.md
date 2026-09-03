

[TOC]

# Introduction

After gaining some basic insights on the capabilities of FDTD simulations on a basic single-lens-system, we are expanding our FDTD simulations for a real telescope SPIDER 2 in the following tutorial. SPIDER2 (Suborbital Polarimeter for Inflation, Dust, and the Epoch of Reionization) was a balloon-borne experiment designed to map the polarization of the CMB. The first flight was taken in January 2015 surveyed roughly 10% of the sky at 95 and 150 GHz. 

`NOTE`: Since its a big system, so its not feasible to run it on the local machine via the jupyter notebook. Here we just discuss the main quantitative and qualitative results that we are extracting from these FDTD simulations. 

So we are going to run this example via the approach mentioned in the [HPC_Tutorial](https://github.com/aa16oaslak/MeepSAT/tree/main/examples/HPC_Tutorial). 

In this Tutorial, we are going to 

- First we define the simulation parameters in `auxiliary_data/02_SPIDER2/SPIDER2.json`.
- Similar to the approach mentioned in the previous tutorial for the single lens system, we write a python script for defining the geometry and running the simulation in `examples/HPC_Tutorial/time-reverse/02_SPIDER2/sim_files/SPIDER2.py`. 
- Finally we define a bash script file for running the time-reverse simulation in `examples/HPC_Tutorial/time-reverse/02_SPIDER2/sim_files/SPIDER2.sh`.

The only difference betweeen time-reverse and time-forward simulations is the source:

- Gaussian source at the focal plane for time-reverse
- Continuous plane waves coming from the sky side at the mentioned frequencies
