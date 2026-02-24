#!/bin/bash
# Submit all BAGEL PT2P evaluation jobs to Slurm
# Time limits are set based on subset size (~40s/sample with 2x safety factor)

NGPU=2
WORK_DIR=/gpfs/home/linjli/source/SpatialReasoning_Eval
OUTPUT_DIR=/gpfs/projects/krishna/linjli/bagel_eval/outputs_bagel_pt2p
LOG_DIR=${WORK_DIR}/slurm_logs_pt2p
CONDA_ENV=/gpfs/projects/krishna/envs/thinkmorph
CPUS=$((NGPU * 8))
MEM=$((NGPU * 200))

# Model download config
export HF_HOME=/gpfs/scrubbed/linjli/hf_cache
MODEL_REPO="ByteDance-Seed/BAGEL-7B-MoT"
MODEL_LOCAL="${HF_HOME}/BAGEL-7B-MoT"

mkdir -p "$LOG_DIR"

# Check if model is already cached; if not, download it once before submitting jobs
if [ -d "$MODEL_LOCAL" ] && [ -f "$MODEL_LOCAL/ema.safetensors" ] && [ -f "$MODEL_LOCAL/ae.safetensors" ]; then
    echo "Model already cached at ${MODEL_LOCAL}"
else
    echo "Model not found in cache. Downloading ${MODEL_REPO} to ${MODEL_LOCAL}..."
    module load conda
    conda activate ${CONDA_ENV}
    export LD_LIBRARY_PATH=${CONDA_ENV}/lib:$LD_LIBRARY_PATH
    huggingface-cli download ${MODEL_REPO} --local-dir ${MODEL_LOCAL}
    if [ $? -ne 0 ]; then
        echo "ERROR: Model download failed. Aborting job submission."
        exit 1
    fi
    echo "Model downloaded successfully."
fi

submit_job() {
    local SUBSET=$1
    local SIDEVIEW=$2  # "" or "_sideview"
    local TIME_LIMIT=$3

    DATASET_NAME="AI2ThorPT2P_${SUBSET}${SIDEVIEW}"
    JOB_NAME="pt2p_${SUBSET}${SIDEVIEW}"

    JID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --qos=normal
#SBATCH --gpus=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out

module load conda
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:\$LD_LIBRARY_PATH
export MASTER_PORT=\$((29500 + RANDOM % 1000))
export HF_HOME=/gpfs/scrubbed/linjli/hf_cache
export THINKMORPH_MODEL_PATH=${MODEL_LOCAL}
export LMUData=/gpfs/scrubbed/linjli/LMUData
export TMPDIR=/tmp/slurm_\${SLURM_JOB_ID}
mkdir -p \$TMPDIR \$LMUData

cd ${WORK_DIR}
torchrun --nproc-per-node=${NGPU} --master_port=\$MASTER_PORT run.py --model bagel_mot --data ${DATASET_NAME} --work-dir ${OUTPUT_DIR}
EOF
    )
    echo "Submitted ${DATASET_NAME}: job ${JID} (time=${TIME_LIMIT})"
}

# Subset sizes and time limits (based on ~40s/sample/GPU, 2x safety, +5min loading)
# dh_midpoint: 162 samples => 2h (ALREADY RUNNING - skip)
# td_midpoint: 242 samples => 3h
# td_ego_dir: 329 samples => 4h
# td_ego_dir_arrow: 337 samples => 4h
# td_ego_side_arrow: 358 samples => 5h
# td_ego_side: 405 samples => 5h
# td_path: 532 samples => 7h
# td_path_arrow: 567 samples => 7h

echo "Submitting all 16 subset jobs..."
echo ""

# dh_midpoint (162 samples)
submit_job "dh_midpoint" "" "02:00:00"
submit_job "dh_midpoint" "_sideview" "02:00:00"

# td_midpoint (242 samples)
submit_job "td_midpoint" "" "03:00:00"
submit_job "td_midpoint" "_sideview" "03:00:00"

# td_ego_dir (329 samples)
submit_job "td_ego_dir" "" "04:00:00"
submit_job "td_ego_dir" "_sideview" "04:00:00"

# td_ego_dir_arrow (337 samples)
submit_job "td_ego_dir_arrow" "" "04:00:00"
submit_job "td_ego_dir_arrow" "_sideview" "04:00:00"

# td_ego_side_arrow (358 samples)
submit_job "td_ego_side_arrow" "" "05:00:00"
submit_job "td_ego_side_arrow" "_sideview" "05:00:00"

# td_ego_side (405 samples)
submit_job "td_ego_side" "" "05:00:00"
submit_job "td_ego_side" "_sideview" "05:00:00"

# td_path (532 samples)
submit_job "td_path" "" "07:00:00"
submit_job "td_path" "_sideview" "07:00:00"

# td_path_arrow (567 samples)
submit_job "td_path_arrow" "" "07:00:00"
submit_job "td_path_arrow" "_sideview" "07:00:00"

echo ""
echo "All 16 jobs submitted (8 subsets x 2 sideview settings)."
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: ${LOG_DIR}"
echo "Results in: ${OUTPUT_DIR}"
