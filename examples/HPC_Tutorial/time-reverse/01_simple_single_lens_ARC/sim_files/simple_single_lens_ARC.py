import sys
import os
import site
from pathlib import Path
import meep as mp
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import json

# Importing the MEEPSAT librarires
import meepsat.simulator as sim
import meepsat.meep_geometry as comp_meep
import meepsat.permittivity_components as comp_eps
import meepsat.stepfunctions as stepfunctions
import meepsat.json_to_script as json_to_script
import meepsat.field_analysis as mpsat_analysis
import meepsat.helpers as mpsat_helpers

# Check if JSON path was provided as command line argument
json_file_path = sys.argv[1]
print(f"Using JSON file from command line argument: {json_file_path}")
data = mpsat_helpers.read_json(json_file_path)

#! Add more parameteters to twick if needed
source_freq, res, savepath_dir, runtime, beam_waist = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
data["sources"]['source1']["frequecy"] = float(source_freq)
data["simulation"]['primary_params']['resolution'] = int(res)
data["output"]["savepath"]["path"] = str(Path(savepath_dir)) + "/" 
# Update the beam waist for the source in the JSON data
data["sources"]['source1']["extra_args"]["width"] = float(beam_waist)  # in mm

#~ ---------------------------------------------

savepath = data["output"]["savepath"]["path"]
os.makedirs(savepath, exist_ok=True)
print('Output directory path:', savepath)

#~ ---------------------------------------------

#~ Initialising MEEPSAT Simulation
cell_X, cell_Y, cell_Z = data["simulation"]['primary_params']['cell_size']['x'], data["simulation"]['primary_params']['cell_size']['y'], data["simulation"]['primary_params']['cell_size']['z'] # Cell Size without considering the PML thickness and its factor


# Initialize the simulation with the different parameters
mpsat_sim = sim.sim_init(sim_name= str(data["simulation"]["name"]),
                        cell_size= [cell_X, cell_Y, cell_Z], # [sx, sy, sz] in mm
                        smallest_freq= data["simulation"]['primary_params']['smallest_freq'], 
                        resolution= data["simulation"]['primary_params']['resolution'],
                        boundary_layer_type= data['boundary_layers']['boundary']['type'],
                        boundary_layer_size= data['boundary_layers']['boundary']['size'],
                        factor_dpml= data['boundary_layers']['boundary']['factor_dpml'])

#~ ---------------------------------------------
# ~ Checking resolution and PML thickness
data, mpsat_sim = sim.check_resolution_and_pml(
    data=data, 
    mpsat_sim=mpsat_sim,
    smallest_freq=data["simulation"]['primary_params']['smallest_freq'],
    highest_n=np.sqrt(5.4)
)


#~ ---------------------------------------------
# Print the simulation parameters
mpsat_sim.print_simulation_parameters()

#~ Adding Sources
source_list = []
exec(json_to_script.source_script(data))
#~ ---------------------------------------------

#~ Adding Boundary
x_left_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.X, side=mp.Low)
x_right_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.X, side=mp.High)
y_down_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.Y, side=mp.Low)
y_up_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.Y, side=mp.High)

custom_boundary_layers = [x_left_boundary, x_right_boundary, y_down_boundary, y_up_boundary]
#~ ---------------------------------------------

#~ Adding Epsilon Map
size_x, size_y, size_z = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1], mpsat_sim.cell_size[2]
res = int(mpsat_sim.resolution)  # Ensure resolution is an integer
# Create the epsilon map: total size of the simulation cell in all the axis multiplied by the resolution + 1
epsilon_map = np.ones((int((size_x)*res+1), 
                       int((size_y)*res+1)), dtype = 'float32')



#~ Adding lens (if given)
exec(json_to_script.add_lens(data))
#~ ---------------------------------------------

#~ Adding box slabs (if given)
exec(json_to_script.add_slab(data))
#~ ---------------------------------------------

#~ Adding aperture (if given)
exec(json_to_script.add_aperture(data))
#~ ---------------------------------------------


#~ DEFINING THE SIMULATION OBJECT
# Mirror symmetry along y direction (x=0 plane)
symmetries = [mp.Mirror(mp.Y, phase=+1)] 

simulation = mp.Simulation(
    cell_size=mpsat_sim.cell,
    sources=source_list,
    resolution=mpsat_sim.resolution,
    boundary_layers=custom_boundary_layers,
    geometry=mpsat_sim.meep_geometry,
    epsilon_input_file = data["output"]["savepath"]["path"] + data["output"]["epsilon_h5_file"]["filename"] +"_epsilon_map" + ".h5",
    symmetries = symmetries,
    force_complex_fields= True)

simulation.use_output_directory(savepath)
#~ ---------------------------------------------

#~ Run simulation briefly to store the epsilon map
sim.plot_and_save_epsilon(
    simulation=simulation,
    savepath=savepath,
    filename_prefix="geometry_plot",
    epsilon_data_name="epsilon",
    size_x=size_x,
    size_y=size_y,
    vmin=0.5,
    vmax=3,
    cmap='viridis',
    figsize=(8, 4),
    dpi=300
)
#~ ---------------------------------------------
#~ Set the stepfunctions parameters
#! Animation Parameters
stepfunctions.set_animation_params(anim_params= {'image_every': data["output"]["animation_options"]["image_every"],
                                              'Nfps': data["output"]["animation_options"]["Nfps"], 
                                              'anim_file_name': savepath + "/"+ data["output"]["animation_options"]["movie_name"] + ".mp4"})

#! Field Parameters
stepfunctions.set_field_params(field_params= {'size_x': size_x,
                                              'size_y': size_y,
                                              'savepath': savepath,
                                              'downsampling_factor_x': data["output"]["animation_options"]["downsample_x"],
                                              'downsampling_factor_y': data["output"]["animation_options"]["downsample_y"],
                                              'method': 'dft'})

#! Runtime parameters
runtime_params = sim.calculate_runtime_parameters(
    source_freq=float(data["sources"]["source1"]["frequecy"]),
    total_time= runtime,
    animation_timestep = data["output"]["animation_options"]["image_every"],
    points_per_period=10,
    extraction_offset=10
)

#~ ---------------------------------------------
stepfunctions.setup_dft_fields(
    simulation, freq=float(data["sources"]["source1"]["frequecy"]))
simulation.run(mp.at_every(runtime_params["animation_timestep"], stepfunctions.Ez2_dB),
               mp.after_time(runtime_params["t0"], mp.at_every(runtime_params["dt"], stepfunctions.count_dft_sample)),
               mp.at_end(stepfunctions.save_animation),
               mp.at_end(stepfunctions.save_dft_fields),
               mp.at_end(stepfunctions.extract_xyzw),
               until = runtime_params["total_time"])

print("Simulation completed.")                                                 

# #~ ---------------------------------------------

# Save the final edited JSON data
with open(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_simulation_data.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Simulation parameters saved to: {data['output']['savepath']['path']}{data['simulation']['name']}_simulation_data.json")
