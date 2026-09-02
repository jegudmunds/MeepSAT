#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#==============================================================================
#~ For MEEPSAT simulations ON your local machine
MEEPSAT_CONDA_ENV="${MEEPSAT_CONDA_ENV:-parallel_meep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MEEPSAT_CONDA_ENV"
#==============================================================================

python_script_file_name="simple_single_lens_ARC"

# file name
file_name="$python_script_file_name"

# output directory + file
base_output_dir="../output_files"
mkdir -p "$base_output_dir"

# JSON file path
json_file_path="./simple_single_lens_ARC.json"

# Define resolution range
resolutions=(12) #20)  # Add your desired resolutions here
runtime=400 # ~Runtime in MEEP units

# Generate a list of frequencies to simulate
freqs=(0.30 0.40 0.50) 

# Beam waist
beam_waist=(1.9434239 1.45756792 1.16605434) # ~ in mm for 90, 120, 150 GHz respectively

# Generate simulation commands for each resolution and frequency combination
for res in "${resolutions[@]}"; do
    for i in "${!freqs[@]}"; do
        freq=${freqs[$i]}
        waist=${beam_waist[$i]}
        real_freq_ghz=$(echo "scale=6; $freq * 300" | bc -l)
        freq_dir_name=$(printf "freq_%.1fGHz" "$real_freq_ghz")
        
        output_dir="${base_output_dir}/${freq_dir_name}/"
        mkdir -p "$output_dir"
        python "$file_name.py" "$json_file_path" "$freq" "$res" "$output_dir" "$runtime" "$waist" > "$output_dir/$file_name.out" 2> "$output_dir/$file_name.err"
        # mpirun -np 2 python "$file_name.py" "$json_file_path" "$freq" "$res" "$output_dir" "$runtime" "$waist"
    done
done
