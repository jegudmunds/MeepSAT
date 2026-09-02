"""
Standalone (non-notebook) test: same original domain/geometry as the very first
validated baseline (cell 160x54->162x56mm, source x=73, lens x=30), but with the
aperture1 / lens1_mount stop plates changed from PEC to a lossy absorbing medium
(n_refr=1.5, D_conductivity=20) instead of mp.perfect_electric_conductor.

Goal: isolate whether the PEC aperture-stop plates -- which by construction extend
all the way out to the cell edge and therefore touch/penetrate the PML -- are the
dominant source of the ~-19dB NTFF contour-independence noise floor, independent of
domain size (a bigger-domain-only test already ruled out PML *distance* as the fix).

Saves to a fresh savepath so neither the original baseline nor the bigcell test
output is touched.
"""
import os
import json
import numpy as np
import meep as mp

import meepsat.simulator as sim
import meepsat.meep_geometry as comp_meep
import meepsat.permittivity_components as comp_eps
import meepsat.stepfunctions as stepfunctions
import meepsat.json_to_script as json_to_script
import meepsat.field_analysis as analysis
import meepsat.helpers as mpsat_helpers

json_file_path = 'auxilary_data/01_simple_single_lens_ARC/simple_single_lens_ARC_absorbing_aperture.json'
data = mpsat_helpers.read_json(json_file_path)

c_mm_s = 299792458.0 * 1000.0
freq = 150.0
a = 1
wvl = c_mm_s / (freq * 1e9)
freq_meep = 1.0 / (wvl * a)
print("freq (meep units):", freq_meep)

beam_waist = 1.1660
data["sources"]["source1"]["frequecy"] = freq_meep
data["sources"]["source1"]["extra_args"]["width"] = beam_waist

savepath = os.path.abspath(f'auxilary_data/01_simple_single_lens_ARC/output_files/{freq}GHz_absorbing_aperture')
os.makedirs(savepath, exist_ok=True)
data["output"]["savepath"]["path"] = savepath + os.sep

cell_X = data["simulation"]['primary_params']['cell_size']['x']
cell_Y = data["simulation"]['primary_params']['cell_size']['y']
cell_Z = data["simulation"]['primary_params']['cell_size']['z']

mpsat_sim = sim.sim_init(sim_name=str(data["simulation"]["name"]),
                          cell_size=[cell_X, cell_Y, cell_Z],
                          smallest_freq=data["simulation"]['primary_params']['smallest_freq'],
                          resolution=data["simulation"]['primary_params']['resolution'],
                          boundary_layer_type=data['boundary_layers']['boundary']['type'],
                          boundary_layer_size=data['boundary_layers']['boundary']['size'],
                          factor_dpml=data['boundary_layers']['boundary']['factor_dpml'])

data, mpsat_sim = sim.check_resolution_and_pml(
    data=data,
    mpsat_sim=mpsat_sim,
    smallest_freq=data["simulation"]['primary_params']['smallest_freq'],
    highest_n=data["lenses"]["lens1"]["n_refr"]
)
mpsat_sim.print_simulation_parameters()

source_list = []
exec(json_to_script.source_script(data))

x_left_boundary = mp.PML(thickness=mpsat_sim.dpml * mpsat_sim.factor_dpml, direction=mp.X, side=mp.Low)
x_right_boundary = mp.PML(thickness=mpsat_sim.dpml * mpsat_sim.factor_dpml, direction=mp.X, side=mp.High)
y_down_boundary = mp.PML(thickness=mpsat_sim.dpml * mpsat_sim.factor_dpml, direction=mp.Y, side=mp.Low)
y_up_boundary = mp.PML(thickness=mpsat_sim.dpml * mpsat_sim.factor_dpml, direction=mp.Y, side=mp.High)
custom_boundary_layers = [x_left_boundary, x_right_boundary, y_down_boundary, y_up_boundary]

size_x, size_y, size_z = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1], mpsat_sim.cell_size[2]
res = int(mpsat_sim.resolution)
epsilon_map = np.ones((int(size_x * res + 1), int(size_y * res + 1)), dtype='float32')

print("DEBUG before add_lens: data[output][savepath][path]=", repr(data["output"]["savepath"]["path"]))
exec(json_to_script.add_lens(data))
exec(json_to_script.add_aperture(data))
print("DEBUG after add_lens/add_aperture: data[output][savepath][path]=", repr(data["output"]["savepath"]["path"]))

epsilon_input_file = data["output"]["savepath"]["path"] + data["output"]["epsilon_h5_file"]["filename"] + "_epsilon_map" + ".h5"
assert os.path.exists(epsilon_input_file), f"epsilon_input_file missing before simulation.run(): {epsilon_input_file}"
print("DEBUG epsilon_input_file exists, size=", os.path.getsize(epsilon_input_file))

symmetries = [mp.Mirror(mp.Y, phase=+1)]

simulation = mp.Simulation(
    cell_size=mpsat_sim.cell,
    sources=source_list,
    resolution=mpsat_sim.resolution,
    boundary_layers=custom_boundary_layers,
    geometry=mpsat_sim.meep_geometry,
    epsilon_input_file=epsilon_input_file,
    symmetries=symmetries,
    force_complex_fields=True)
simulation.use_output_directory(savepath)

stepfunctions.set_animation_params(anim_params={
    'image_every': data["output"]["animation_options"]["image_every"],
    'Nfps': data["output"]["animation_options"]["Nfps"],
    'anim_file_name': savepath + "/" + data["output"]["animation_options"]["movie_name"] + ".mp4"})
stepfunctions.set_field_params(field_params={
    'size_x': size_x, 'size_y': size_y, 'savepath': savepath,
    'downsampling_factor_x': data["output"]["animation_options"]["downsample_x"],
    'downsampling_factor_y': data["output"]["animation_options"]["downsample_y"]})

runtime = 600
runtime_params = sim.calculate_runtime_parameters(
    source_freq=float(data["sources"]["source1"]["frequecy"]),
    resolution=mpsat_sim.resolution,
    steady_state_time=runtime,
    courant=simulation.Courant,
    min_periods_for_steady_state=10,
    periods_to_average=4,
    points_per_period=10,
    animation_timestep=data["output"]["animation_options"]["image_every"])

os.makedirs(savepath, exist_ok=True)  # defensive: meep's use_output_directory can trash existing dirs
print("DEBUG right before simulation.run(): savepath exists?", os.path.isdir(savepath))

simulation.run(mp.at_every(runtime_params["animation_timestep"], stepfunctions.Ez2_dB),
               mp.after_time(runtime_params["t0"], mp.at_every(runtime_params["dt"], stepfunctions.accumulate_efield_and_hfield)),
               mp.at_end(stepfunctions.save_animation),
               mp.at_end(stepfunctions.save_accumulated_fields),
               mp.at_end(stepfunctions.extract_xyzw),
               until=runtime_params["total_time"])
print("Simulation completed.")

with open(data["output"]["savepath"]["path"] + "/" + data["simulation"]["name"] + "_simulation_data.json", "w") as f:
    json.dump(data, f, indent=2)

# ---- Post-processing: same box as the ORIGINAL PEC baseline for a clean A/B test ----
c = 2.998e+11
wvl_meep = c / (freq * 1e9)

ntff_box = (-76.0, 78.0, -24.0, 24.0)
ntff_box_inner = (-74.0, 76.0, -23.0, 23.0)
ntff_angles_deg = np.linspace(-90, 90, 3601)

noise_floor_dB, _, _ = analysis.noise_floor_db(savepath, wvl_meep, ntff_angles_deg, ntff_box, ntff_box_inner)
within_20deg = np.abs(ntff_angles_deg) <= 20
print(f"[absorbing aperture test] NTFF noise floor within +/-20deg (p90): {np.percentile(noise_floor_dB[within_20deg], 90):.1f} dB")
print(f"[absorbing aperture test] NTFF noise floor full +/-90deg (p90): {np.percentile(noise_floor_dB, 90):.1f} dB")
print("baseline (original PEC aperture, same box): -18.7 / -20.0 dB")
