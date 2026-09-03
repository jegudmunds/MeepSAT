"""
This module contains all the necessary functions for initialising the MeepSAT simulation
"""

import math
import os
import warnings
from typing import Callable
#import meep_testings as mp
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
import h5py
import meepsat.meep_geometry as comp 
import meepsat.permittivity_components as comp_eps 
import meepsat.helpers as exf 


def calculate_runtime_parameters(source_freq, resolution, steady_state_time, animation_timestep,
                                 courant=0.5, min_periods_for_steady_state=10, periods_to_average=4,
                              points_per_period=10):
    """
    Calculate exact runtime parameters compatible with MEEP's discrete timestep.
    Parameters:
    -----------
    source_freq : float
        Source frequency in MEEP units (dimensionless, c/a)
    resolution : int
        MEEP grid resolution (pixels per unit distance)
    courant : float, optional
        The Courant factor used by MEEP (default: 0.5)
    min_periods_for_steady_state : int, optional
        Number of periods to let the wave propagate before starting extraction
    periods_to_average : int, optional
        Number of full optical cycles to average fields over (must be an integer)
    points_per_period : int, optional
        Target number of sampling points per wave period
    Returns:
    --------
    dict
        Validated parameters snapped to MEEP's internal grid clocks.
    """
    import math
    
    # 1. Calculate fundamental continuous variables
    period = 1.0 / source_freq
    meep_internal_dt = courant / resolution  # Actual FDTD time step
    # 2. Fix the sampling interval (dt) to match MEEP's internal clock
    target_dt = period / points_per_period
    # Calculate how many internal MEEP timesteps fit into our target sampling dt
    steps_per_sample = max(1, round(target_dt / meep_internal_dt))
    # True discrete dt that MEEP will actually use
    actual_dt = steps_per_sample * meep_internal_dt
    actual_points_per_period = period / actual_dt
    # 3. Determine when to start sampling
    if steady_state_time is not None:
        # Snap user-provided steady-state time to the FDTD timestep
        t0 = math.ceil(steady_state_time / meep_internal_dt) * meep_internal_dt
    else:
        # Estimate from the requested number of periods
        t0 = math.ceil(
            (min_periods_for_steady_state * period) / meep_internal_dt
        ) * meep_internal_dt
        
    # 4. Enforce extraction window to be an EXACT integer multiple of the period
    extraction_window = periods_to_average * period
    total_time = t0 + extraction_window
    
    runtime_params = {
        'period': period,
        'meep_internal_dt': meep_internal_dt,
        'dt': actual_dt,                         # Use this in meep.at_every(dt, ...)
        't0': t0,                                 # Start averaging here
        'total_time': total_time,                 # Stop simulation here
        'points_per_period': actual_points_per_period,
        'steps_per_sample': steps_per_sample,
        'animation_timestep': animation_timestep
    }
    print(f"Validated MEEP Runtime Parameters:")
    print(f"  Wave Period:                 {period:.4f} time units")
    print(f"  MEEP Internal Timestep:      {meep_internal_dt:.6f} time units")
    print(f"  Snapped Sampling Step (dt):  {actual_dt:.6f} time units (Every {steps_per_sample} FDTD steps)")
    print(f"  Actual Points Per Period:    {actual_points_per_period:.2f}")
    print(f"  Steady State Delay (t0):     {t0:.4f} time units")
    print(f"  Total Run Time (total_time): {total_time:.4f} time units")
    print(f"  Extraction Window Size:      {extraction_window:.4f} time units ({periods_to_average} full cycles)")
    print(f"  Animation Timestep:          {animation_timestep:.4f} time units")
    return runtime_params


def plot_and_save_epsilon(simulation, savepath, filename_prefix, epsilon_data_name, 
                          size_x, size_y, vmin=0.5, vmax=3, cmap='viridis', show_plot= False,
                          figsize=(8, 4), dpi=300, return_epsilon=False, focalplane_x=None,
                          plot_pml=True, pml_thickness=None, pml_color='red', pml_alpha=0.2,
                          save_h5= True):
    """
    Plot and save the epsilon (permittivity) map from a MEEP simulation.
    
    Parameters:
    -----------
    simulation : mp.Simulation
        The MEEP simulation object
    savepath : str
        Directory path where files will be saved
    filename_prefix : str
        Prefix for the output filenames (e.g., "geometry_plot")
    epsilon_data_name : str
        Name for the epsilon dataset in the HDF5 file
    size_x : float
        Size of simulation cell in x direction (mm)
    size_y : float
        Size of simulation cell in y direction (mm)
    vmin : float, optional
        Minimum value for colormap scale (default: 0.5)
    vmax : float, optional
        Maximum value for colormap scale (default: 3)
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (8, 4))
    dpi : int, optional
        Resolution for saved figure (default: 300)
    plot_pml : bool, optional
        Whether to plot PML boundary layers (default: True)
    pml_thickness : float, optional
        Thickness of PML layers. If None, extracted from simulation
    pml_color : str, optional
        Color for PML regions (default: 'red')
    pml_alpha : float, optional
        Transparency of PML regions (default: 0.2)
    focalplane_x : float, optional
        X-coordinate of focal plane to mark (default: None)
    return_epsilon : bool, optional
        Whether to return the epsilon array (default: False)
    save_h5 : bool, optional
        Whether to save the epsilon data in an HDF5 file (default: True)
    
    Returns:
    --------
    epsilon : np.ndarray (optional)
        The extracted epsilon array if return_epsilon=True
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Run simulation briefly to get epsilon
    simulation.run(until=0)
    epsilon = simulation.get_epsilon()
    
    # Extract PML thickness if not provided
    if plot_pml and pml_thickness is None:
        # Try to get PML thickness from simulation
        if hasattr(simulation, 'boundary_layers') and simulation.boundary_layers:
            pml_thickness = simulation.boundary_layers[0].thickness
        else:
            print("Warning: Could not determine PML thickness. Set plot_pml=False or provide pml_thickness.")
            plot_pml = False
    
    # Plot the epsilon map geometry
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(epsilon.T, interpolation='spline36', cmap=cmap, origin='lower', 
                   extent=[-size_x/2, size_x/2, -size_y/2, size_y/2],
                   vmin=vmin, vmax=vmax)
    
    # Plot PML boundary layers
    if plot_pml and pml_thickness is not None:
        # Left PML
        ax.axvspan(-size_x/2, -size_x/2 + pml_thickness, 
                   alpha=pml_alpha, color=pml_color, label='PML', zorder=10)
        # Right PML
        ax.axvspan(size_x/2 - pml_thickness, size_x/2, 
                   alpha=pml_alpha, color=pml_color, zorder=10)
        # Bottom PML
        ax.axhspan(-size_y/2, -size_y/2 + pml_thickness, 
                   alpha=pml_alpha, color=pml_color, zorder=10)
        # Top PML
        ax.axhspan(size_y/2 - pml_thickness, size_y/2, 
                   alpha=pml_alpha, color=pml_color, zorder=10)
    
    # Plot focal plane if provided
    if focalplane_x is not None:
        ax.axvline(x=focalplane_x, color='blue', linestyle='--', 
                   linewidth=2, label='Focal Plane')
    
    # Add legend if there are labeled elements
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')
    
    # plt.colorbar(im, ax=ax, label='Permittivity (ε)')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Permittivity (ε)')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Epsilon Map with Boundary Layers')
    plt.savefig(os.path.join(savepath, f"{filename_prefix}.png"), dpi=dpi, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    if save_h5:
        # Save the epsilon map to an HDF5 file
        h5_filename = os.path.join(savepath, f"{filename_prefix}.h5")
        with h5py.File(h5_filename, "w") as h5file:
            h5file.create_dataset(epsilon_data_name, data=epsilon)
        
        print(f"Epsilon plot saved to: {os.path.join(savepath, filename_prefix)}.png")
        print(f"Epsilon data saved to: {h5_filename}")
    
    if return_epsilon:
        return epsilon


def check_resolution_and_pml(data,
                             mpsat_sim,
                             meep_sim = None,
                             smallest_freq: float = None,
                             highest_n: float = None,
                             smallest_length: float = None):
    """
    Function to check if the resolution and pml are sufficient for the simulation

    Arguments
    ---------
    data : dict
        Dictionary containing the simulation parameters (extracted from the json file)
    meep_sim: meep Simulation object
        MEEP simulation object
    highest_freq : float
        Highest frequency of the source in meep units
    highest_n : float
        Highest refractive index of the materials in the simulation box
    smallest_length: float = None
        Smallest length scale in the simulation box (e.g., smallest feature size)

    Returns
    -------
    data : dict
        Updated dictionary containing the simulation parameters (extracted from the json file)
    """
    
    
    if meep_sim is None and highest_n is None:
        raise ValueError("Either the MEEP simulation object or the highest refractive index must be provided to check the resolution and PML thickness.")
    
    # Initialize arrays at the beginning to avoid UnboundLocalError
    arc_length_arr = []
    arc_n_arr = []
    
    #! 0th Step: Check the smallest length scale (wavelength) and convert it into frequency. 
    if smallest_length is not None:
        largest_freq = 1/smallest_length
        if data["simulation"]['primary_params']['resolution'] / largest_freq < 8:
            warnings.warn(f"Resolution criteria for smallest_length not met. Required at least 8 points per wavelength, but got {data['simulation']['primary_params']['resolution'] / largest_freq}. Increasing resolution.")
            data["simulation"]['primary_params']['resolution'] = int(largest_freq * 8)
 
    #! 1ST step: Extract the highest refractive index from the epsilon data
    if highest_n is None:
        #! 0TH step: Run the simulation to extract the epsilon data
        print("No highest refractive index provided. Extracting the highest refractive index from the epsilon data...")
        print("Running a quick simulation to extract the epsilon data for checking the resolution and PML thickness...")
        meep_sim.run(until=0)
        print("Simulation run complete!")
        print("Extracting the epsilon data...")
        epsilon_map = meep_sim.get_epsilon() 
        highest_n = np.sqrt(np.max(epsilon_map))
        print("Highest refractive index in the simulation box: ", highest_n)
    else:
        print("Highest refractive index provided: ", highest_n)

    #! Checking the PML thickness and factor
    given_pml_thickness = data['boundary_layers']['boundary']['size']*data['boundary_layers']['boundary']['factor_dpml']
    
    # Print statements for old PML and resolution values
    print("Given PML thickness: ", given_pml_thickness)
    print("Given resolution: ", data["simulation"]['primary_params']['resolution'])

    #! Checking the source frequency and assuming it to be the largest frequency present in the system
    if "sources" in data:
        print("Assuming the wavelength of the Source to be the largest wavelength present in the system and doing a sanity check on the PML thickness.")
        for source in data["sources"]:
            wavelength_meep_source = 1 / data["sources"][source]["frequency"]  # Wavelength in MEEP unit
            min_pml_thickness = 0.5*wavelength_meep_source # Source: https://meep-hr.readthedocs.io/en/latest/FAQ/#checking-convergence
            if given_pml_thickness < min_pml_thickness:
                print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
                data['boundary_layers']['boundary']['size'] = min_pml_thickness

    if smallest_freq is not None:
        print("Smallest frequency provided: ", smallest_freq)
        wavelength_meep_largest = 1 / smallest_freq  # Wavelength in MEEP unit
        # It should at least 1/2 wavelength_meep_ of the source
        min_pml_thickness = 0.5*wavelength_meep_largest # Source: https://meep-hr.readthedocs.io/en/latest/FAQ/#checking-convergence
        if given_pml_thickness < min_pml_thickness:
            print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
            data['boundary_layers']['boundary']['size'] = min_pml_thickness

    if highest_n is not None:
        print("Highest refractive index provided: ", highest_n)
        if "sources" in data:
            for source in data["sources"]:
                wavelength_meep_ = 1 / data["sources"][source]["frequency"]  # Wavelength in MEEP unit
                wavelength_meep_inside_medium = wavelength_meep_ / highest_n  # Wavelength in MEEP unit inside the medium
                min_pml_thickness = 0.5*wavelength_meep_inside_medium # Source: https://meep-hr.readthedocs.io/en/latest/FAQ/#checking-convergence
                if given_pml_thickness < min_pml_thickness:
                    print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
                    data['boundary_layers']['boundary']['size'] = min_pml_thickness

    

    #! Check if the resolution criteria is met or not for the highest refractive index
    if highest_n is not None:
        print("Highest refractive index provided: ", highest_n)
        wavelength_meep_ = 1 / data["sources"]['source1']["frequency"]  # Wavelength in MEEP unit
        wavelength_meep_inside_medium = wavelength_meep_ / highest_n  # Wavelength in MEEP unit inside the medium
        freq_inside_medium = 1 / wavelength_meep_inside_medium  # Frequency inside the medium
        if data["simulation"]['primary_params']['resolution'] / freq_inside_medium < 8:
            print("Resolution criteria doesn't meet the criteria for the smallest frequency. Increasing the resolution to meet the criteria.")
            print("Wavelength Meep: ", wavelength_meep_)
            print("Wavelength Meep inside medium: ", wavelength_meep_inside_medium)
            data["simulation"]['primary_params']['resolution'] = int(freq_inside_medium * 8)

    #! Checking if the resolution criteria is met or not for the source's frequency
    # resolution/frequency ratio should be at least 10   
    if "sources" in data:
        print("Assuming the wavelength of the Source to be the largest wavelength present in the system and doing a sanity check on the PML thickness.")
        for source in data["sources"]:
            if data["simulation"]['primary_params']['resolution'] / data["sources"][source]["frequency"] < 10:
                print("Resolution criteria doesn't meet the criteria for the provided source frequency. Increasing the resolution to meet the criteria.")
                data["simulation"]['primary_params']['resolution'] = int(data["sources"][source]["frequency"] * 8)

    #! Checking if the resolution criteria is met or not for the smallest frequency
    if smallest_freq is not None:
        print("Smallest frequency provided: ", smallest_freq)
        if data["simulation"]['primary_params']['resolution'] / smallest_freq < 10:
            print("Resolution criteria doesn't meet the criteria for the smallest frequency. Increasing the resolution to meet the criteria.")
            data["simulation"]['primary_params']['resolution'] = int(smallest_freq * 8)

    #! Now assuming the smallest frequency present in the system is the ARC thickness (lambda/4 assumption)
    # check if data["lenses"] exists in data
    if "lenses" in data:
        #! Important: THE BELOW TWO LIST WILL CONTAIN THE VALUES FOR ALL THE LENGTH SCALES AND REFRACTIVE INDICES PRESEENT IN THE DIFFERENT LENSES 
        arc_length_arr = []; 
        arc_n_arr = []
        for lens in data["lenses"]:
            # Checking for #! Single ARC parameters
            if "AR_left" in data["lenses"][lens] or "AR_right" in data["lenses"][lens]:
                arc_length_arr.append(data["lenses"][lens]["AR_left"])
                arc_length_arr.append(data["lenses"][lens]["AR_right"])
                arc_n_arr.append(data["lenses"][lens]["AR_material"])

            # Checking for #! Multi-layer ARC parameters
            if "AR_left_layers" in data["lenses"][lens]:
                arc_length_arr.extend(data["lenses"][lens]["AR_left_layers"])
                arc_n_arr.extend(data["lenses"][lens]["AR_left_materials"])
            if "AR_right_layers" in data["lenses"][lens]:
                arc_length_arr.extend(data["lenses"][lens]["AR_right_layers"])
                arc_n_arr.extend(data["lenses"][lens]["AR_right_materials"])


            # Checking for #! Stepped pyramid ARC parameters
            if "ARC_type" in data["lenses"][lens]:
                if data["lenses"][lens]["ARC_type"] == "stepped_pyramid":
                    # Since pitch is a single float value and kerf, width are lists
                    arc_length_arr.append(data["lenses"][lens]["step_ARC_pitch"])  # Remove the brackets
                    arc_length_arr.extend(data["lenses"][lens]["step_ARC_kerf"])
                    arc_length_arr.extend(data["lenses"][lens]["step_ARC_depth"])
                    # Appending the materials
                    # Check if step_ARC_material_nref is a list or a single value
                    material_nref = data["lenses"][lens]["step_ARC_material_nref"]
                    if isinstance(material_nref, list):
                        arc_n_arr.extend(material_nref)
                    else:
                        # If it's a single float value, append it instead of extend
                        arc_n_arr.append(material_nref)

            # Checking for #! Delamination layer parameters
            if "delam_thick" in data["lenses"][lens]:
                if data["lenses"][lens]["delam_thick"] != 0:
                    arc_length_arr.append(data["lenses"][lens]["delam_thick"])
                    arc_length_arr.append(data["lenses"][lens]["delam_width"])

            # Checking for #! Surface error parameters
            #!! UPDATE THIS LATER WHEN YOU ARE USING SURFACE ERROR IN THE SIMULATION

        # ! Now checking the resolution criteria for all the arc_n_arr 
        for n in arc_n_arr:
            wavelength_meep_inside_medium = wavelength_meep_ / n  # Wavelength in MEEP unit inside the medium
            freq_inside_medium = 1 / wavelength_meep_inside_medium  # Frequency inside the medium
            if data["simulation"]['primary_params']['resolution'] / freq_inside_medium < 8:
                print("Resolution criteria doesn't meet the criteria for the ARC layers. Increasing the resolution to meet the criteria.")
                data["simulation"]['primary_params']['resolution'] = int(freq_inside_medium * 8)

        #! Checking the resolution criteria for all the arc_length_arr
        for wavelength_meep_arc in arc_length_arr:
            # Add validation to ensure wavelength_meep_arc is a scalar
            if isinstance(wavelength_meep_arc, (list, tuple)):
                # If it's accidentally a list, extract the first element or flatten
                if len(wavelength_meep_arc) > 0:
                    wavelength_meep_arc = wavelength_meep_arc[0]
                else:
                    continue
            
            if wavelength_meep_arc == 0:
                print(f"Warning: Found zero wavelength in arc_length_arr, skipping...")
                continue
                
            freq_arc = 1 / wavelength_meep_arc  # Frequency corresponding to the ARC layer thickness
            if data["simulation"]['primary_params']['resolution'] / freq_arc < 8:
                print("Resolution criteria doesn't meet the criteria for the ARC layers. Increasing the resolution to meet the criteria.")
                data["simulation"]['primary_params']['resolution'] = int(freq_arc * 8)



    mpsat_sim.resolution = data["simulation"]['primary_params']['resolution']
    
    print("All length scales of lenses in the simulation: ", arc_length_arr)
    print("All refractive indices of different components in the of lense in the simulation: ", arc_n_arr)
    print("Modified resolution: ", data["simulation"]['primary_params']['resolution'])
    print("Modified PML thickness: ", data['boundary_layers']['boundary']['size']*data['boundary_layers']['boundary']['factor_dpml'])

    return data, mpsat_sim

def convert_to_meep_units(self, value, unit_type, from_unit='um'):
    """
    Converts real-world units to MEEP simulation units.
    
    In MEEP, the simulation uses normalized units where c=1. This function helps
    convert from physical units to MEEP's normalized units based on a chosen
    length scale.
    
    Parameters:
    ----------
    value : float
        The numerical value to convert
    unit_type : str
        The type of unit being converted:
        - 'length': Length units (e.g., μm to MEEP units)
        - 'frequency': Frequency units (e.g., THz to MEEP units)
        - 'time': Time units (e.g., ps to MEEP units)
    from_unit : str, optional
        The physical unit to convert from. Default is 'um' (micrometers).
        Supported units:
        - Length: 'nm', 'um', 'mm', 'm'
        - Frequency: 'Hz', 'GHz', 'THz'
        - Time: 'fs', 'ps', 'ns', 's'
    
    Returns:
    -------
    float
        The converted value in MEEP units
    
    Examples:
    --------
    # Convert 1.55 μm wavelength to MEEP units
    wavelength_meep = convert_to_meep_units(1.55, 'length', 'um')
    
    # Convert 193 THz frequency to MEEP units
    freq_meep = convert_to_meep_units(193, 'frequency', 'THz')
    
    # Convert 100 fs time to MEEP units
    time_meep = convert_to_meep_units(100, 'time', 'fs')
    """
    # Speed of light in m/s
    c = 299792458

    # Define the base length unit (default is μm)
    length_scale = 1.0  # MEEP units

    # Length conversion factors to meters
    length_to_meters = {
        'nm': 1e-9,
        'um': 1e-6,
        'mm': 1e-3,
        'm': 1.0
    }
    
    # Frequency conversion factors to Hz
    freq_to_hz = {
        'Hz': 1.0,
        'GHz': 1e9,
        'THz': 1e12
    }
    
    # Time conversion factors to seconds
    time_to_seconds = {
        'fs': 1e-15,
        'ps': 1e-12,
        'ns': 1e-9,
        's': 1.0
    }
    
    if unit_type == 'length':
        if from_unit not in length_to_meters:
            raise ValueError(f"Unsupported length unit: {from_unit}. Use one of {list(length_to_meters.keys())}")
        # Convert length to MEEP units (normalized to length_scale)
        return value * length_to_meters[from_unit] / (length_to_meters['um'] * length_scale)
    
    elif unit_type == 'frequency':
        if from_unit not in freq_to_hz:
            raise ValueError(f"Unsupported frequency unit: {from_unit}. Use one of {list(freq_to_hz.keys())}")
        # Convert frequency to MEEP units
        frequency_hz = value * freq_to_hz[from_unit]
        wavelength_m = c / frequency_hz
        return length_to_meters['um'] * length_scale / wavelength_m
    
    elif unit_type == 'time':
        if from_unit not in time_to_seconds:
            raise ValueError(f"Unsupported time unit: {from_unit}. Use one of {list(time_to_seconds.keys())}")
        # Convert time to MEEP units
        time_s = value * time_to_seconds[from_unit]
        return time_s * c / (length_to_meters['um'] * length_scale)
    
    else:
        raise ValueError("Unit type must be 'length', 'frequency', or 'time'")

class sim_init():
    """
    For initialising the simulation parameters
    """

    def __init__(self,
                 sim_name: str = None,
                 cell_size: list = None,
                 smallest_freq: float = None,
                 smallest_wavelength: float = None,
                 resolution: float = None,
                 boundary_layer_type: str = None,
                 boundary_layer_size: float = None,
                 factor_dpml: float = None,
                 verbosity: int = 0):
        
        """ 
        Initialises the simulation parameters

        Arguments
        ---------
        sim_name : str
            Name of the simulation

        cell_size : list
            Size of the cell in the x, y and z directions
            For 2D simulations, the z-direction size is 0 and the cell size is [sx, sy, 0]
            For 3D simulations, the cell size is [sx, sy, sz]
            `Current supported options: Only 2D`

        freq : float
            Frequency of the source in meep units

        wavelength : float
            Wavelength of the source in meep units
            If the frequency is provided, the wavelength is calculated as 1/freq (c=1)

        resolution : float
            Resolution of the simulation: number of grid points per unit meep wavelength

        boundary_layer_type : str 
            Type of the boundary layer
            Three basic types of terminations are supported in Meep: 
            Bloch-periodic boundaries, metallic walls, and PML absorbing layers
            `Curent supported options: 'PML'

        boundary_layer_size : float
            Thickness of the boundary layer
            `Current supported options: Only PML (dpml)`

        factor_dpml : float
            Factor by which the boundary layer thickness is multiplied 
            `Current supported options available: Only for PML` 
        """        
        self.sim_name = sim_name
        #==================================
        if smallest_freq:
            self.freq = smallest_freq
            self.wavelength = 1/smallest_freq
        elif smallest_wavelength:
            self.wavelength = smallest_wavelength
            self.freq = 1/smallest_wavelength
        else:
            raise ValueError('Either frequency or wavelength should be provided')
        
        #==================================

        # if resolution/self.freq < 8:
        #     raise ValueError('The resolution should be atleast 8 points per wavelength. The grid size should be small enough that it can accurately resolve the wavelength of the electromagnetic wave, but not too small to unnecessarily increase computational requirements.')
        # else:
        self.resolution = resolution

        #==================================
        if boundary_layer_type == 'PML':
            self.boundary_layer_type = boundary_layer_type
            self.dpml = boundary_layer_size

            if factor_dpml:
                self.factor_dpml = factor_dpml
            else:
                warnings.warn('No factor provided for the PML boundary layer thickness. Assuming the default factor to be 2')
                self.factor_dpml = 2

            self.cell_size = [cell_size[0] + self.factor_dpml*self.dpml, cell_size[1] + self.factor_dpml*self.dpml, 0]
            self.cell = mp.Vector3(self.cell_size[0], self.cell_size[1], self.cell_size[2])
        #==================================
        ### Here we can add other boundary layer types in the future versions
        # elif boundary_layer_type == 'metallic':
        #================================== 
        else:
            raise ValueError('Only PML boundary layer is supported in the current MEEPSAT version')
        
        # ! REST IS IMP: BUT THESE TWO ARE VERY IMPORTANT PARAMETERS FOR THE SIMULATION
        self.meep_geometry = [] # List to store the optical components made using the MEEP functions

        #! The below is a useless placeholder: Remove it in the future
        self.eps_geometry = [] # List to store the optical components made using the epsilon functions

    def print_simulation_parameters(self):
        """
        Prints the simulation parameters including the simulation name, cell size, frequency, wavelength, resolution, 
        boundary layer type, boundary layer size, and the factor for PML boundary layer thickness.
        Parameters:
        None
        Returns:
        None
        """
        print(f"Simulation name: {self.sim_name}")
        print(f"Cell size: {self.cell_size}")
        print(f"Frequency: {self.freq}")
        print(f"Wavelength: {round(self.wavelength, 2)}")
        print(f"Resolution: {self.resolution}")
        print(f"Boundary layer type: {self.boundary_layer_type}")
        print(f"Boundary layer size: {self.dpml}")
        print(f"Factor for PML boundary layer thickness: {self.factor_dpml}")


    def list_components(self):
        """
        Prints the components of the MEEP and Epsilon geometries.
        This method prints the components of the `meep_geometry` and `eps_geometry`
        attributes of the class instance. It first prints the components of the 
        `meep_geometry` followed by the components of the `eps_geometry`.
        Output:
            Prints the components to the console.
        """
        
        print('---MEEP Function Components---')
        for component in self.meep_geometry:
            print(component)
        print('----------------\n')
        
        print('---Epsilon Function Components---')
        for component in self.eps_geometry:
            print(component)
        print('----------------')

    def add_meep_geometry(self, object):
        """
        Adds the MEEP objects/components to the simulation

        Arguments
        ---------
        object : object
            Object created using the MEEP functions
            can be any of the following:
            list of MEEP objects, individual MEEP objects
        """
        self.meep_geometry.append(object)
        print("{} added to the list of components created using the MEEP functions!".format(object))

    def add_eps_geometry(self,
                          component: object = None):
        """
        Adding the eps component to the component list

        Arguments
        ---------
        component : object
            Component created using the epsilon functions in the components_2D_eps.py file
        """
        self.eps_geometry.append(component)
        print("{} added to the list of components created using the epsilon functions!".format(component))

