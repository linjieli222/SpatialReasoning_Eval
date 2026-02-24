#!/bin/bash
# Submit all 16 BAGEL PT2P evaluation jobs to Slurm
# Each subset runs as a separate job using data parallelism across GPUs
#
# Usage:
#   bash slurm_bagel_pt2p.sh          # default 2 GPUs per job
#   bash slurm_bagel_pt2p.sh 4        # 4 GPUs per job

NGPU=${1:-2}

WORK_DIR=/gpfs/home/linjli/source/SpatialReasoning_Eval
OUTPUT_DIR=/gpfs/projects/krishna/linjli/bagel_eval/outputs_bagel_pt2p
LOG_DIR=${WORK_DIR}/slurm_logs_pt2p
CONDA_ENV=/gpfs/projects/krishna/envs/thinkmorph

mkdir -p "$LOG_DIR"

SUBSETS=(
    dh_midpoint
    td_ego_dir
    td_ego_dir_arrow
    td_ego_side
    td_ego_side_arrow
    td_midpoint
    td_path
    td_path_arrow
)

CPUS=$((NGPU * 8))
MEM=$((NGPU * 200))

for SUBSET in "${SUBSETS[@]}"; do
    for SIDEVIEW in "" "_sideview"; do
        DATASET_NAME="AI2ThorPT2P_${SUBSET}${SIDEVIEW}"
        JOB_NAME="pt2p_${SUBSET}${SIDEVIEW}"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --qos=normal
#SBATCH --gpus=${NGPU}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out

module load conda
conda activate ${CONDA_ENV}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:\$LD_LIBRARY_PATH
export MASTER_PORT=\$((29500 + RANDOM % 1000))
export HF_HOME=/gpfs/scrubbed/linjli/hf_cache
export LMUData=/gpfs/scrubbed/linjli/LMUData
mkdir -p \$LMUData

cd ${WORK_DIR}
torchrun --nproc-per-node=${NGPU} --master_port=\$MASTER_PORT run.py --model bagel_mot --data ${DATASET_NAME} --work-dir ${OUTPUT_DIR}
EOF

        echo "Submitted: ${DATASET_NAME} (${NGPU} GPUs)"
    done
done

echo ""
echo "All 16 jobs submitted with ${NGPU} GPUs each."
echo "Monitor with: squeue -u \$USER"
echo "Logs in: ${LOG_DIR}"
echo "Results in: ${OUTPUT_DIR}"
