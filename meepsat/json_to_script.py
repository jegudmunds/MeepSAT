import json
import os

#! THE FOLLOWING FUNCTION MIGHT CAUSE ERRORS IF THE PATHS ARE NOT SET CORRECTLY


def temporary_meepasat_path_set(json_data=None):
    # Check if savepath is provided in JSON data
    savepath_from_json = ""
    if json_data and "output" in json_data and "savepath" in json_data["output"]:
        savepath_from_json = json_data["output"]["savepath"]["path"]
    
    script = f"""
import sys
import os
import site
from pathlib import Path

# First check for MEEPSAT_PATH environment variable (highest priority)
meepsat_env_path = os.environ.get('MEEPSAT_PATH')
if meepsat_env_path and os.path.exists(os.path.join(meepsat_env_path, 'meepsat')):
    print(f"Using MEEPSAT from environment variable: {{meepsat_env_path}}")
    main_dir = meepsat_env_path
    meepsat_dir = os.path.join(main_dir, 'meepsat')
    sys.path.append(main_dir)
    sys.path.append(meepsat_dir)
    meepsat_found = True
else:
    # Method 1: Try to import directly if MEEPSAT is installed properly
    try:
        import meepsat
        print("MEEPSAT found in installed packages")
        meepsat_dir = os.path.dirname(meepsat.__file__)
        main_dir = os.path.dirname(meepsat_dir)
        sys.path.append(meepsat_dir)  # Add meepsat directory to path for submodule imports
        meepsat_found = True
    except ImportError:
        # Method 2: Try to find MEEPSAT in common locations
        possible_paths = [
            Path.cwd().parent,                     # Parent of current directory
            Path.cwd(),                           # Current directory
            Path(__file__).resolve().parent.parent,  # Parent of script directory
            Path.home() / "Phd_work/MEEPSAT_WFH",      # User's home directory + common path
            Path('/cfs/data/asab1238/MEEPSAT_WFH'),   # HPC specific path
            Path('/cfs/data/asab1238/MEEPSAT_WFH')    # Another HPC specific path
        ]
        
        meepsat_found = False
        for path in possible_paths:
            try:
                meepsat_dir = path / "meepsat"
                if meepsat_dir.exists():
                    main_dir = str(path)
                    meepsat_dir = str(meepsat_dir)
                    sys.path.append(main_dir)
                    sys.path.append(meepsat_dir)  # Critical: add meepsat directory for submodule imports
                    print(f"MEEPSAT found at: {{meepsat_dir}}")
                    meepsat_found = True
                    break
            except Exception as e:
                print(f"Error checking path {{path}}: {{e}}")
        
        if not meepsat_found:
            print("WARNING: MEEPSAT directory not found automatically.")
            print("Please set the MEEPSAT_PATH environment variable before running.")
            print("For example: export MEEPSAT_PATH=/path/to/MEEPSAT")
            main_dir = "."
            meepsat_dir = "./meepsat"
            sys.path.append(main_dir)
            sys.path.append(meepsat_dir)


# Handle savepath - use JSON path if provided, otherwise create default
"""
    
    # Rest of your function remains the same...
    if savepath_from_json:
        script += f"""
# Using savepath from JSON configuration
savepath = '{savepath_from_json}'
os.makedirs(savepath, exist_ok=True)
"""
    else:
        script += f"""
# Creating default output directory in parent directory
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output_files')
os.makedirs(output_dir, exist_ok=True)
savepath = output_dir + '/'
"""

    script += f"""
# Fix imports by making sure Python can find the meepsat submodules directly
sys.path.insert(0, meepsat_dir)  # Prioritize the meepsat directory

print('Main directory path:', main_dir)
print('MEEPSAT library path:', meepsat_dir)
print('Output directory path:', savepath)
print('Python path now includes:', sys.path[:5])  # Show the first 5 paths
"""
    return script

#! <-- THE FUNCTION ENDS HERE --> #

def import_modules():
    script = f"""
import meep as mp
import numpy as np
import h5py
import matplotlib.pyplot as plt
    """
    return script


def import_meepsat_modules():
    script = f"""
# Importing the MEEPSAT librarires
import meepsat.simulation_2D as sim
import meepsat.components_2D_meep as comp_meep
import meepsat.components_2D_eps as comp_eps
import meepsat.visualization as mpsat_plot
import meepsat.mp_sim_run as mp_sim_run
"""
    return script

def init_meepsat_sim(data):
    script = f"""
#~ Initialising MEEPSAT Simulation
cell_X, cell_Y, cell_Z = {data["simulation"]['primary_params']['cell_size']['x']}, {data["simulation"]['primary_params']['cell_size']['y']}, {data["simulation"]['primary_params']['cell_size']['z']} # Cell Size without considering the PML thickness and its factor

# Initialize the simulation with the different parameters
mpsat_sim = sim.sim_init(sim_name='{data["simulation"]["name"]}',
                        cell_size= [cell_X, cell_Y, cell_Z], # [sx, sy, sz] in mm
                        smallest_freq= {data["simulation"]['primary_params']['smallest_freq']}, 
                        resolution= {data["simulation"]['primary_params']['resolution']},
                        boundary_layer_type= '{data['boundary_layers']['boundary']['type']}',
                        boundary_layer_size= {data['boundary_layers']['boundary']['size']},
                        factor_dpml= {data['boundary_layers']['boundary']['factor_dpml']})

# Print the simulation parameters
mpsat_sim.print_simulation_parameters()
"""
    return script

#! <-- VARIOUS SOURCE SCRIPT SEGMENTS --> #
def GaussianBeam(source_data):
    # Handle the special mp.inf case
    if source_data["extra_args"]["end_time"] == "mp.inf":
        end_time = "mp.inf"
    else:
        end_time = source_data["extra_args"]["end_time"]
        
    script = f"""
#~ Adding Source: GaussianBeam
gaussian_source_init = comp_meep.GaussianBeam(mpsat_sim=mpsat_sim,
                                              center= mp.Vector3({source_data["center_x"]},
                                                                {source_data["center_y"]},
                                                                {source_data["center_z"]}),
                                              size= mp.Vector3({source_data["size_x"]},
                                                              {source_data["size_y"]},
                                                              {source_data["size_z"]}),
                                              component= '{source_data["component"]}',
                                              freq= {source_data["frequency"]},
                                              angle= {source_data["extra_args"]["angle"]},
                                              width= {source_data["extra_args"]["width"]},
                                              kwargs= {{'beam_x0': mp.Vector3({source_data["extra_args"]["beam_x0"]["x"]},
                                                                             {source_data["extra_args"]["beam_x0"]["y"]},
                                                                             {source_data["extra_args"]["beam_x0"]["z"]}),
                                                       'beam_E0': mp.Vector3({source_data["extra_args"]["beam_E0"]["x"]},
                                                                             {source_data["extra_args"]["beam_E0"]["y"]},
                                                                             {source_data["extra_args"]["beam_E0"]["z"]}),
                                                       'start_time': {source_data["extra_args"]["start_time"]},
                                                       'end_time': {end_time}}})
source_list.append(gaussian_source_init.assemble())
"""
    return script

def continuous_planewaves(data):
    script = f"""
#~ Adding Source: ContinuousPlaneWaves  
cpw_source_init = comp_meep.ContinuousPlaneWaves(mpsat_sim=mpsat_sim,
                                                center= mp.Vector3({data["center_x"]},
                                                                  {data["center_y"]},
                                                                  {data["center_z"]}),
                                                size= mp.Vector3({data["size_x"]},
                                                                {data["size_y"]},
                                                                {data["size_z"]}),
                                                component= '{data["component"]}',
                                                freq= {data["frequency"]},
                                                angle= {data["extra_args"]["angle"]},
                                                rot_axis= '{data["extra_args"].get("rot_axis", "z")}',
                                                kwargs= {{"is_integrated": {data["extra_args"]["is_integrated"]},
                                                         "start_time": {data["extra_args"]["start_time"]},
                                                         "end_time": {data["extra_args"]["end_time"]}}})
source_list.append(cpw_source_init.assemble())
"""
    return script
    
def source_script(data):
    script = f"""source_list = []"""
    
    for source in data["sources"]:
        source_data = data["sources"][source]
        # Check the type of source
        if source_data['type'] == 'GaussianBeam':
            script = script + '\n' + GaussianBeam(source_data)
        elif source_data['type'] == 'ContinuousPlaneWaves':
            script = script + '\n' + continuous_planewaves(source_data)
        else:
            raise ValueError(f"Source type {source_data['type']} not recognized.")
    
    return script

#! <-- THE END: VARIOUS SOURCE SCRIPT SEGMENTS --> #
#^====================================================================================================================^#

#! <-- BOUNDARY SCRIPT SEGMENTS --> #
def PML_boundary(data):
    script = f"""
#~ Adding Boundary: PML
# PML Layers
boundary = comp_meep.Boundary(type='PML',
                            thickness= mpsat_sim.dpml)

# Creating an MEEP object for the boundary using the assemble function
pml = boundary.assemble()
"""
    return script

def boundary_script(data):
    for boundary in data["boundary_layers"]:
        boundary_data = data["boundary_layers"][boundary]
        if boundary_data["type"] == "PML":
            script = PML_boundary(boundary_data)
        else:
            raise ValueError(f"Boundary type {boundary_data['type']} not recognized.")
    return script
#! <-- THE END: BOUNDARY SCRIPT SEGMENTS --> #

#^====================================================================================================================^#
def init_epsilon_map(data):
    script= f"""
#~ Adding Epsilon Map
size_x, size_y, size_z = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1], mpsat_sim.cell_size[2]
res = int(mpsat_sim.resolution)  # Ensure resolution is an integer
# Create the epsilon map: total size of the simulation cell in all the axis multiplied by the resolution + 1
epsilon_map = np.ones((int((size_x)*res+1), 
                       int((size_y)*res+1)), dtype = 'float32')
"""
    return script

#^====================================================================================================================^#
def convert_string_to_object_reference(value, param_name=None):
    """
    Converts string references to Python object references for the script.
    
    If value is a string that starts with specific prefixes like 'mp.',
    it will be returned without quotes so it can be evaluated as a Python expression.
    
    Args:
        value: The value to potentially convert
        param_name: Optional parameter name to exclude from conversion
        
    Returns:
        The value appropriately formatted for Python code
    """
    # Parameters that should always remain as strings
    string_params = ["name", "lens_type", "ARC_type", "step_ARC_angle", "step_ARC_rot_axis"]
    
    # If this is a parameter that should always be a string, return it quoted
    if param_name in string_params:
        return f"'{value}'"
    
    # Special handling for specific string values that should remain as strings
    string_values = ["default", "stepped_pyramid", "perpendicular_to_surface", "extended_aspheric", "aspheric"]
    
    if isinstance(value, str):
        # Check if the value is in our list of string values that should stay quoted
        if value in string_values:
            return f"'{value}'"
            
        # Prefixes that indicate the string should be treated as a Python object reference
        reference_prefixes = ["mp.", "np.", "None"]
        
        # Check if the string starts with any of our reference prefixes
        for prefix in reference_prefixes:
            if value.startswith(prefix):
                return value  # Return without quotes so it's interpreted as a reference
                
        # If it doesn't match any prefix pattern, return with quotes
        return f"'{value}'"
    
    # For non-string values, return as-is
    return value


def add_lens(data):
    if "lenses" not in data:
        return ""  # Empty string since the lens is not a mandatory component
    else:
        script = ""
        for lens in data["lenses"]:
            lens_data = data["lenses"][lens]
            
            # Build a dictionary of parameters with appropriate conversions
            params = {}
            for key, value in lens_data.items():
                params[key] = convert_string_to_object_reference(value, key)
            
            # Check if this lens has multi-layer AR coating parameters
            has_multi_arc = ('AR_left_layers' in lens_data or 'AR_right_layers' in lens_data)
            
            # Check if this lens has stepped pyramid ARC coating parameters
            has_stepped_pyramid_arc = ('ARC_type' in lens_data and lens_data['ARC_type'] == 'stepped_pyramid')

            script += f"""
#~ Adding Lens: {lens}
{lens}_init = comp_eps.AsphericLens(name = {params.get('name', "'lens'")},
                                r1 = {params.get('r1', 'np.inf')},
                                r2 = {params.get('r2', 'np.inf')},
                                c1 = {params.get('c1', 0)},
                                c2 = {params.get('c2', 0)},
                                lens_type = {params.get('lens_type', "'extended_aspheric'")},
                                a1_coeffs = {params.get('a1_coeffs', 'None')},
                                a2_coeffs = {params.get('a2_coeffs', 'None')},
                                thick = {params['thick']},
                                diameter = {params['diameter']},
                                x = {params['x']},
                                n_refr= {params['n_refr']},
                                AR_left= {params.get('AR_left', 'None')},
                                AR_right= {params.get('AR_right', 'None')},
                                AR_material= {params.get('AR_material', 'None')},
                                AR_left_layers={params.get('AR_left_layers', 'None')},
                                AR_left_materials={params.get('AR_left_materials', 'None')},
                                AR_right_layers={params.get('AR_right_layers', 'None')},
                                AR_right_materials={params.get('AR_right_materials', 'None')},
                                ARC_type= {params.get('ARC_type', "'default'")},
                                step_ARC_nlayers= {params.get('step_ARC_nlayers', 'None')},
                                step_ARC_pitch= {params.get('step_ARC_pitch', 'None')},
                                step_ARC_kerf= {params.get('step_ARC_kerf', 'None')},
                                step_ARC_depth= {params.get('step_ARC_depth', 'None')},
                                step_ARC_width= {params.get('step_ARC_width', 'None')},
                                step_ARC_material= {params.get('step_ARC_material', 'None')},
                                step_ARC_angle= {params.get('step_ARC_angle', "'perpendicular_to_surface'")},
                                step_ARC_rot_axis= {params.get('step_ARC_rot_axis', "'z'")},
                                step_ARC_offset= {params.get('step_ARC_offset', [0,0])},
                                eps= epsilon_map,
                                mpsat_sim= mpsat_sim)
"""
            # Rest of the function remains the same...
            if has_multi_arc:
                script += f"""
# Using multi-layer AR coating assembly
{lens} = {lens}_init.assemble_with_multi_arc()
"""
            elif has_stepped_pyramid_arc:
                script += f"""
# Using stepped pyramid ARC coating assembly
{lens},  {lens}_stepped_pyramid_ARC_blocks = {lens}_init.assemble_with_stepped_pyramid_ARC()
# Adding the list of stepped pyramid ARC MEEP blocks to meepsat geometry
for stepped_pyramid_ARC_blocks in {lens}_stepped_pyramid_ARC_blocks:
    mpsat_sim.
    (stepped_pyramid_ARC_blocks)
""" 
            else:
                script += f"""
# Using standard assembly with single-layer AR coating
{lens} = {lens}_init.assemble()
"""

            script += f"""
mpsat_sim.add_eps_geometry({lens})
# Plot the lens
# {lens}_init.plot_lenses()
"""
        
        # Check if we need to save the epsilon map
        save_epsilon = "epsilon_h5_file" in data.get("output", {})

        # Add code to save epsilon map only if needed
        if save_epsilon:
            print("Saving epsilon map to HDF5 file...")
            script += f"""
#~ Saving the Epsilon Map
# Saving the epsilon_map initiated after the final to a .h5 file, so that meep can read it
{lens}_init.write_h5file(filename= '{data["output"]["savepath"]["path"]}' + '{data["output"]["epsilon_h5_file"]["filename"]}_epsilon_map',
                        parallel= {data["simulation"]["parallel"]})
"""
        return script
    


def square_aperture(aperture_data, aperture_name):
    # Handle the special mp.inf case for conductivity
    if aperture_data["conductivity"] == "mp.inf":
        conductivity = "mp.inf"
    else:
        conductivity = aperture_data["conductivity"]
    
    # Determine which position parameter(s) to use
    pos_x = aperture_data.get("pos_x", None)
    pos_y = aperture_data.get("pos_y", None)
    orientation = aperture_data.get("orientation", None)

    if pos_x is None and pos_y is None:
        raise ValueError(f"Aperture {aperture_name} must have either pos_x or pos_y defined")

    if pos_x is not None and pos_y is not None and orientation is None:
        raise ValueError(f"Aperture {aperture_name} defines both pos_x and pos_y, "
                         "so it must also define the orientation "
                         "('vertical' or 'horizontal')")

    # Build position parameter string (both positions can be given together)
    pos_params = []
    if pos_x is not None:
        pos_params.append(f"pos_x={pos_x}")
    if pos_y is not None:
        pos_params.append(f"pos_y={pos_y}")
    if orientation is not None:
        pos_params.append(f"orientation='{orientation}'")
    pos_param = ",\n                                       ".join(pos_params)

    # Absorber layers on the faces of the stop, as a list of dicts that
    # ApertureStop turns into ApertureAbsorberLayer objects
    absorber_layers = aperture_data.get("absorber_layers", None)

    script = f"""
#~ Adding Aperture: Square
{aperture_name}_init = comp_meep.ApertureStop(mpsat_sim=mpsat_sim,
                                       type='{aperture_data["type"]}',
                                       diameter={aperture_data["diameter"]},
                                       thickness={aperture_data["thickness"]},
                                       {pos_param},
                                       n_refr={aperture_data.get("n_refr", 1.0)},
                                       conductivity={conductivity},
                                       material={aperture_data.get("material", "None")},
                                       y_centre_offset={aperture_data.get("y_centre_offset", [0, 0])},
                                       y_size_offset={aperture_data.get("y_size_offset", [0, 0])},
                                       print_vertices={aperture_data.get("print_vertices", False)},
                                       absorber_layers={absorber_layers},
                                       preserve_aperture={aperture_data.get("preserve_aperture", False)}
)

{aperture_name}_up, {aperture_name}_down = {aperture_name}_init.assemble()
mpsat_sim.add_meep_geometry({aperture_name}_up)
mpsat_sim.add_meep_geometry({aperture_name}_down)
"""

    if absorber_layers:
        script += f"""
# Absorber layers on the faces of {aperture_name}
{aperture_name}_absorbers = {aperture_name}_init.get_absorbers()
mpsat_sim.meep_geometry.extend({aperture_name}_absorbers)
"""

    return script

def add_aperture(data):
    script = ""
    # First, check if the apertures are present in the data
    if "apertures" not in data:
        return script # Empty string since the aperture is not an mandatory component
    else:
        for aperture in data["apertures"]:
            aperture_data = data["apertures"][aperture]
            if aperture_data["type"] == "square":
                script += square_aperture(aperture_data, aperture_name=aperture)
            else:
                raise ValueError(f"Aperture type {aperture_data['type']} not recognized.")
        return script

#^====================================================================================================================^#
def add_slab(data):
    """
    Create code for adding slabs defined in the JSON file
    
    Parameters
    ----------
    data : dict
        JSON data containing slab definitions
        
    Returns
    -------
    script : str
        Python script code for creating slabs
    """
    script = ""
    
    # Check if slabs are present in the data
    if "slabs" not in data:
        return script  # Empty string since slabs are not mandatory
    else:
        print("Slab found in the JSON data")
        script += "\n#~ Adding Slabs\n"
        
        for slab_name, slab_data in data["slabs"].items():
            # Handle material specially - it might be a complex expression
            material_str = slab_data.get("material", "None")
            
            # Process center and size parameters - they might be strings like "mp.Vector3(...)"
            center_str = slab_data.get("center", "None")
            size_str = slab_data.get("size", "None")
            
            script += f"""
# Slab: {slab_name}
{slab_name}_init = comp_meep.Slab(
    mpsat_sim=mpsat_sim,
    name='{slab_data.get("name", "block")}',
    center={center_str},
    size={size_str},
    material={material_str},
    angle={slab_data.get("angle", 0)},
    rot_axis='{slab_data.get("rot_axis", "x")}'
)
{slab_name}_obj = {slab_name}_init.assemble()
mpsat_sim.add_meep_geometry({slab_name}_obj)
"""
        return script

#^====================================================================================================================^#
def add_absorbers(data):
    """
    Create code for adding absorbers defined in the JSON file
    
    Parameters
    ----------
    data : dict
        JSON data containing absorber definitions
        
    Returns
    -------
    script : str
        Python script code for creating absorbers
    """
    script = ""
    
    # Check if absorbers are present in the data
    if "absorbers" not in data:
        return script  # Empty string since absorbers are not mandatory
    else:
        print("Absorber found in the JSON data")
        script += "\n#~ Adding Absorbers\n"
        
        for absorber_name, absorber_data in data["absorbers"].items():
            # Build a dictionary of parameters with appropriate conversions
            params = {}
            for key, value in absorber_data.items():
                params[key] = convert_string_to_object_reference(value, key)

            # Deal with special absorber parameters if needed
            #!) freq
            if "freq" in absorber_data:
                print(f"Absorber frequency found: {absorber_data['freq']}")
                params["freq"] = convert_string_to_object_reference(absorber_data["freq"], "freq")
            else:
                print(f"No frequency specified for absorber, using the current source frequency: {data['sources']['source1']['frequency']}")
                params["freq"] = data["sources"]["source1"]["frequency"]

            
            script += f"""
# Absorber: {absorber_name}
{absorber_name}_init = comp_meep.PyramidalAbsorbers(
    mpsat_sim=mpsat_sim,
    base_width={params.get('base_width', 'None')},
    top_width={params.get('top_width', 0)},
    height={params.get('height', 'None')},
    layer_thickness={params.get('layer_thickness', 'None')},
    num_pyramids={params.get('num_pyramids', 20)},
    n_layers={params.get('n_layers', 10)},
    material={params.get('material', 'None')},
    epsilon_real={params.get('epsilon_real', 2.5)},
    epsilon_imag={params.get('epsilon_imag', 0)},
    freq={params.get('freq', '1/3')},
    x_coverage_start={params.get('x_coverage_start', 'None')},
    x_coverage_end={params.get('x_coverage_end', 'None')},
    y_coverage_start={params.get('y_coverage_start', 'None')},
    y_coverage_end={params.get('y_coverage_end', 'None')},
    coverage_percent={params.get('coverage_percent', 40)},
    edges={params.get('edges', '["top", "bottom"]')},
    x_left_offset={params.get('x_left_offset', 0)},
    x_right_offset={params.get('x_right_offset', 0)},
    y_top_offset={params.get('y_top_offset', 0)},
    y_bottom_offset={params.get('y_bottom_offset', 0)},
    add_substrate={params.get('add_substrate', 'True')},
    substrate_thickness={params.get('substrate_thickness', 'None')},
    substrate_material={params.get('substrate_material', 'None')},
    substrate_epsilon_real={params.get('substrate_epsilon_real', 'None')},
    substrate_epsilon_imag={params.get('substrate_epsilon_imag', 'None')},
    substrate_extends_beyond_pyramids={params.get('substrate_extends_beyond_pyramids', 'True')},
    substrate_extension={params.get('substrate_extension', 0.5)},
    add_pec_backing={params.get('add_pec_backing', 'False')},
    pec_thickness={params.get('pec_thickness', 'None')},
    pec_extends_beyond_substrate={params.get('pec_extends_beyond_substrate', 'True')},
    pec_extension={params.get('pec_extension', 0)},
    name={params.get('name', 'None')}
)

{absorber_name}_blocks = {absorber_name}_init.assemble()
# Add all absorber blocks to the MEEP geometry
mpsat_sim.meep_geometry.extend({absorber_name}_blocks)
"""
        return script

#^====================================================================================================================^#

# Add after the existing at_end_command_for_volume_monitors function

def at_every_command_for_flux_monitors(at_every_list, timestep):
    """
    Generate at_every commands specifically for flux monitors
    """
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        # Flux monitor functions are directly in mp_sim_run
        command += f"mp.at_every({timestep}, mp_sim_run.{data}),\n"

    return command.rstrip(",\n")  # Remove trailing comma and newline if present      

def at_end_command_for_flux_monitors(at_end_list):
    """
    Generate at_end commands specifically for flux monitors
    """
    if not at_end_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_end_list:
        if data == "calculate_transmission_reflection":
            # Use mp_sim_run prefix for flux calculations
            command += f"mp.at_end(mp_sim_run.{data}),\n"
        else:
            command += f"mp.at_end(mp_sim_run.{data}),\n"

    return command.rstrip(",\n")  # Remove trailing comma and newline if present


def add_flux_monitor(data):
    """
    Create code for adding flux monitors defined in the JSON file
    """
    script = ""
    
    # Check if flux monitors are present in the data
    if "flux_monitor" not in data:
        # Set flux_monitor_list to None
        script += "\nflux_monitor_list=None\n"
        return script  # Empty string since flux monitors are not mandatory
    else:
        script += "\n#~ Adding Flux Monitors\nflux_monitor_list = []\n"
        
        for monitor_name, monitor_data in data["flux_monitor"].items():
            # Special handling for Python objects in the JSON
            center = monitor_data["center"]
            size = monitor_data["size"]
            direction = monitor_data.get("direction", "mp.X")
            monitor_type = monitor_data.get("monitor_type", "transmission")
            freq_min = monitor_data.get("freq_min", "None")
            freq_max = monitor_data.get("freq_max", "None")
            nfreq = monitor_data.get("nfreq", 100)
            
            # Add support for using saved flux files
            use_flux_file = monitor_data.get("use_flux_file", "None")
            
            script += f"""
# Flux Monitor: {monitor_name}
{monitor_name}_init = comp_meep.FluxMonitor(
    mpsat_sim=mpsat_sim,
    name='{monitor_name}',
    center={center},
    size={size},
    direction={direction},
    freq_min={freq_min},
    freq_max={freq_max},
    nfreq={nfreq},
    monitor_type='{monitor_type}',
    use_flux_file={use_flux_file}
)
{monitor_name}_region = {monitor_name}_init.assemble()
flux_monitor_list.append({{"{monitor_name}": [{monitor_data}, {monitor_name}_region]}})
"""
        
        # Add code to initialize the flux monitor registry
        script += "\n# Initialize the flux monitor registry\n"
        script += "mp_sim_run.set_flux_monitor_registry(flux_monitor_list, "
        
        # Add directory if specified
        if "output" in data and "flux_monitor_data_dir" in data["output"]:
            script += f"monitor_data_save_dir='{data['output']['flux_monitor_data_dir']}', "
        else:
            script += "monitor_data_save_dir=savepath + 'flux_monitor_data/', "
        
        # Add save frequency if specified
        if "output" in data and "flux_monitor_save_freq" in data["output"]:
            script += f"monitor_data_save_freq={data['output']['flux_monitor_save_freq']}, "
        else:
            script += "monitor_data_save_freq=5, "
        
        # Add reference simulation flag
        script += f"is_reference={data.get('run_reference_sim', False)}"
        
        script += ")\n"
        
        return script

#^====================================================================================================================^#
def init_meep_sim_object(data):
    """
    Create the MEEP simulation object with optional epsilon map handling
    """
    # Base parameters that are always included
    params = [
        "cell_size=mpsat_sim.cell",
        "boundary_layers=[pml]",
        "geometry=mpsat_sim.meep_geometry",
        "sources=source_list"
    ]
    
    # Add epsilon_h5_file parameter only if specified and if lenses are present
    # (since epsilon maps are primarily used with lenses)
    has_lenses = "lenses" in data
    has_epsilon_file = "epsilon_h5_file" in data.get("output", {})
    
    if has_lenses and has_epsilon_file:
        params.append(f"epsilon_h5_file=savepath + '{data['output']['epsilon_h5_file']['filename']}_epsilon_map'")
    
    # Join parameters with commas and proper indentation
    params_str = ",\n                            ".join(params)
    
    # Build the script
    script = f"""
# Initialize the meep simulation object with the different parameters
sim = mpsat_sim.meep_sim_obj({params_str})
"""
    return script

#^====================================================================================================================^#
def add_required_data(data):
    script = f"""
#~ Runtime of the simulation
run_time = {data["output"]["runtime"]}
#~ Requested data by the user
data_required= {data["output"]["data_required"]}
"""
    return script
#^====================================================================================================================^#
def add_monitor(data):
    """
    Create code for adding monitors defined in the JSON file
    
    Parameters
    ----------
    data : dict
        JSON data containing monitor definitions
        
    Returns
    -------
    script : str
        Python script code for creating monitors
    """
    script = ""
    
    # Check if monitors are present in the data
    if "monitor" not in data:
        return script  # Empty string since monitors are not mandatory
    else:
        script += "\n#~ Adding Monitors\nmonitor_list = []\n"
        
        for monitor_name, monitor_data in data["monitor"].items():
            # Special handling for Python objects in the JSON
            center = monitor_data["center"]
            size = monitor_data["size"]
            script += f"""
# Monitor: {monitor_name}
{monitor_name}_init = comp_meep.VolumeMonitor(mpsat_sim=mpsat_sim,
                                            name='{monitor_name}',
                                            center={center},
                                            size={size},
                                            data_required={monitor_data["data_required"]})
{monitor_name}_obj = {monitor_name}_init.assemble()
monitor_list.append({{"{monitor_name}": [{monitor_data}, {monitor_name}_obj]}})
"""
        # script += "\nmp_sim_run.set_volume_monitor_registry(monitor_list)\n" # set the volume monitor registry
        
        return script
    

#! THIS IS WORKING MATE BUT WITHOUT THE MEMORY EFFICIENCY CODE FOR ANIMATION:
def run_meepsat_sim(data):
    # Extract the plotting params if they exist
    plotting_params = "None"
    if "plotting_params" in data["output"]["animation_options"]:
        plotting_params = json.dumps(data["output"]["animation_options"]["plotting_params"])
    
    script = f"""
mp_sim_run.set_animation_params(anim_params= {{'image_every': {data["output"]["animation_options"]["image_every"]}, 
                                             'Nfps': {data["output"]["animation_options"]["Nfps"]}, 
                                             'anim_file_name': savepath + '{data["output"]["animation_options"]["movie_name"]}',
                                             'plotting_params': {plotting_params}}})

mpsat_sim.run_simulation(sim = sim,
                         mpsat_sim = mpsat_sim,
                         runtime = run_time,
                         get_mp4 = {str(data["output"]["animation_options"]["get_mp4"]).capitalize()},
                         image_every = {data["output"]["animation_options"]["image_every"]},
                         Nfps = {data["output"]["animation_options"]["Nfps"]},
                         savepath = savepath,
                         movie_name = '{data["output"]["animation_options"]["movie_name"]}',
                         requested_data = data_required,
                         monitor_list = monitor_list if 'monitor_list' in locals() else None,
                         save_data = False)
"""
    return script


#^====================================================================================================================^#
### JSON TO SCRIPT CONVERSION ###
def json_to_pyscript(json_file, output_dir=None, output_name=None):
    data = read_json(json_file)
    
    # Collect all script components
    script_components = [
        # Your existing components
        temporary_meepasat_path_set(data),
        import_modules(),
        import_meepsat_modules(),
        init_meepsat_sim(data),
        source_script(data),
        boundary_script(data),
        init_epsilon_map(data),
        add_aperture(data),
        add_slab(data),
        add_lens(data),
        add_monitor(data),
        add_flux_monitor(data),
        #! Add other components here..
        init_meep_sim_object(data),
        add_required_data(data),
        run_meepsat_sim(data)
    ]
    
    # Join with newlines between components
    script = "\n\n".join(script_components)
    
    # Determine output filename and path
    if output_name:
        script_name = output_name
    else:
        json_basename = os.path.basename(json_file)
        script_name = os.path.splitext(json_basename)[0] + "_meepsat_script.py"
    
    # Determine the output path
    if output_dir:
        output_path = os.path.join(output_dir, script_name)
    else:
        output_path = os.path.join("example_scripts", script_name)
    
    write_script(script, output_path)
    return output_path

def read_json(json_file):
    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def write_script(script, script_file):
    # Write the script to a Python file
    with open(script_file, "w") as f:
        f.write(script)
    print(f"Script written to {script_file}")
    # return script_file

#! <-- THE END: JSON TO SCRIPT CONVERSION --> #

#! <-- THE FOLLOWING FUNCTIONS ARE USED IN simulation_2D.py to generate the sim.run script --> #
#^====================================================================================================================^#
def at_every_command(at_every_list, timestep):
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        command += f"mp.at_every({timestep}, mp_sim_run.{data}),\n"
    
    return command.rstrip(",\n")  # Remove trailing comma and newline if present



def at_end_command(at_end_list):
    if not at_end_list:  # Handle empty list case
        return ""
        
    command = ""
    for data in at_end_list:
        if data == "save_animation":
            command += f"mp.at_end(mp_sim_run.save_animation),\n"
        elif data == "calculate_transmission_reflection":
            command += f"mp.at_end(mp_sim_run.calculate_transmission_reflection),\n"  # Use mp_sim_run prefix
        else:
            command += f"mp.at_end(mp.{data}),\n"
    
    return command.rstrip(",\n")  # Remove trailing comma and newline if present

def data_required_script_inside_sim_run(data_required):
    script = ""
    script += at_every_command(data_required["at_every"], "image_every")
    # Add a comma and newline 
    script += ",\n"
    script += at_end_command(data_required["at_end"])
    return script
#^====================================================================================================================^#
#! <-- FUNCTIONS TO HANDLE MONITORS INSIDE sim.run --> #

def at_every_command_for_volume_monitors(at_every_list, timestep, monitor_type):
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        if monitor_type == "mp.Volume":
            command += f"mp.at_every({timestep}, mp_sim_run.{data}_VolumeMonitor),\n"
        # Add other monitor types here
        else:
            raise ValueError(f"Monitor type {monitor_type} not recognized.")  

    return command.rstrip(",\n")  # Remove trailing comma and newline if present      

def at_end_command_for_volume_monitors(at_end_list, monitor_type=None):
    if not at_end_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_end_list:
        if data.startswith("output_"):
            command += f"mp.at_end({data}),\n"
        elif monitor_type == "mp.Volume":
            command += f"mp.at_end(mp_sim_run.{data}_VolumeMonitor),\n"
        # Add other monitor types here
        else:
            raise ValueError(f"Monitor type {monitor_type} not recognized.")
        
    return command.rstrip(",\n")  # Remove trailing comma and newline if present

def data_required_inside_indiv_monitors(monitor_list):
    script = ""
    for monitor_entry in monitor_list:
        # Each entry is a dict with one key (monitor name)
        monitor_name = list(monitor_entry.keys())[0]
        monitor_content = monitor_entry[monitor_name]
        # [{'monitor_1': [{'details': 'here'}, <monitor_object>]}] 
        # monitor_content is a list where:
        # monitor_content[0] is the monitor data dictionary 
        # monitor_content[1] is the monitor object
        monitor_data = monitor_content[0]
        

        # Print the entire monitor data structure for debugging
        print(f"Monitor data structure for {monitor_name}:", monitor_data)
        
        # Check if "at_every" exists before accessing
        at_every_list = monitor_data.get("data_required", {}).get("at_every", [])
        at_every_timestep = monitor_data.get("data_required", {}).get("at_every_timestep", 0)
        print(f"at_every list for {monitor_name}:", at_every_list)

        # Add at_every commands if they exist
        if at_every_list:
            script += at_every_command_for_volume_monitors(
                at_every_list, 
                at_every_timestep,
                monitor_data.get("type", None)  # Default to None if not specified
            )
            # Add a comma and newline
            script += ",\n"
        
        # Check if "at_end" exists before accessing
        at_end_list = monitor_data.get("data_required", {}).get("at_end", [])
        if at_end_list:
            script += at_end_command_for_volume_monitors(at_end_list, monitor_data.get("type", None))
    
    return script.rstrip(",\n")  # Remove trailing comma and newline if present

#! <-- THE END: FUNCTIONS TO HANDLE MONITORS INSIDE sim.run --> #
#^====================================================================================================================^#
#! EDIT THE BELOW FUNCTION TO ADD MORE SIM.RUN SCRIPT SEGMENTS IN FUTURE (IF REQUIRED):
def sims_data_requested(data_required, run_time, monitor_list):
    # Prepare the content for the run function separately
    run_args = data_required_script_inside_sim_run(data_required)
    if monitor_list is not None:
        run_args += ",\n"
        monitor_run_args = data_required_inside_indiv_monitors(monitor_list)
        run_args += monitor_run_args

    # If run_args is not empty, add a comma between commands
    if run_args:
        run_args += ",\n"
    run_args += run_time["command"]

    # Apply indentation to each line of run_args
    indented_args = "\n".join(f"    {line}" for line in run_args.split("\n"))
    
    main_script = f"""self.sim.run(
{indented_args}
)"""
    # print(main_script)
    return main_script
