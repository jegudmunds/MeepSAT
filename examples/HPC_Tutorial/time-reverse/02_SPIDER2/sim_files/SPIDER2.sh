#!/usr/bin/env bash
#SBATCH --job-name=SPIDER2_simulations
#SBATCH --partition=cops              # Partition name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=8            # 8 CPU cores per array task
#SBATCH --mem=0                      # Request all available memory per node
#SBATCH --time=100:00:00             # Time limit hrs:min:sec
#SBATCH --array=0-2                  # Run 3 array tasks (one per frequency: indices 0, 1, 2)
#SBATCH --output=job_%A_%a.log  # %A=job ID, %a=array index

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#==============================================================================
# Activating the conda parallel MEEP environment in the HPC cluster
MEEPSAT_CONDA_ENV="${MEEPSAT_CONDA_ENV:-parallel_meep}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MEEPSAT_CONDA_ENV"
#==============================================================================

# # Load the singularity module
# module load singularity

# # Define the path to the Singularity image
# MEEP_SINGULARITY_IMAGE=/cfs/data/asab1238/singularity_container/meepsat.sif
#==============================================================================

python_script_file_name="SPIDER2"

# file name
file_name="$python_script_file_name"

# output directory + file
base_output_dir="../output_files"
mkdir -p "$base_output_dir"

# JSON file path
json_file_path="./SPIDER2.json"

# Define resolution range
resolutions=(12) # Add your desired resolutions here
runtime=4000 # ~Runtime in MEEP units

# Generate a list of frequencies to simulate
freqs=(0.30 0.40 0.50) 

# Beam waist
beam_waist=(1.9434239 1.45756792 1.16605434) # ~ in mm for 90, 120, 150 GHz respectively

echo "======================================"
echo "Starting SPIDER2 simulations"
echo "Resolutions: ${resolutions[@]}"
echo "Frequencies: ${freqs[@]}"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "======================================"

# Use SLURM_ARRAY_TASK_ID to select which frequency to run
for res in "${resolutions[@]}"; do
    i="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job.}"
    freq=${freqs[$i]}
    waist=${beam_waist[$i]}
    real_freq_ghz=$(echo "scale=6; $freq * 300" | bc -l)
    freq_dir_name=$(printf "freq_%.1fGHz" "$real_freq_ghz")
    
    output_dir="${base_output_dir}/${freq_dir_name}/"
    mkdir -p "$output_dir"
    
    echo "[INFO] Running frequency: $real_freq_ghz GHz (array task $SLURM_ARRAY_TASK_ID)"
    echo "======================================"
    
    mpirun -np "${SLURM_CPUS_PER_TASK:-8}" python "$file_name.py" "$json_file_path" "$freq" "$res" "$output_dir" "$runtime" "$waist" > "$output_dir/$file_name.out" 2> "$output_dir/$file_name.err"
    
    # Using singularity to run the Python script inside the container for MEEP single digit precision
    # Bind with the actual filesystem path (not symlink)
    # singularity exec \
    # --bind /cfs/data:/cfs/data \
    # --bind /cfs/data1:/cfs/data1 \
    # $MEEP_SINGULARITY_IMAGE bash -c "source /meep/venv/bin/activate && mpirun -np 8 python $file_name.py $json_file_path $freq $res $output_dir $runtime $waist" \
    # > $output_dir/$file_name.out 2> $output_dir/$file_name.err


    echo "[INFO] Simulation completed for frequency: $real_freq_ghz GHz"
    echo "======================================"
done
