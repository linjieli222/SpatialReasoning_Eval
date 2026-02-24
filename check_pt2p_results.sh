#!/bin/bash
# Check status and results for all BAGEL PT2P evaluation jobs
WORK_DIR=/gpfs/home/linjli/source/SpatialReasoning_Eval
OUTPUT_DIR=${WORK_DIR}/outputs_bagel_pt2p
LOG_DIR=${WORK_DIR}/slurm_logs_pt2p

echo "=== BAGEL PT2P Evaluation Status ==="
echo "Date: $(date)"
echo ""

echo "=== Running Jobs ==="
squeue -u linjli --format="%.10i %.20j %.2t %.10M %.5D %R" 2>/dev/null
echo ""

echo "=== Completed Results ==="
for f in $(find ${OUTPUT_DIR} -name "*_acc.csv" 2>/dev/null | sort); do
    echo "--- $(basename $f) ---"
    cat "$f"
    echo ""
done

echo "=== Job Completion Summary ==="
SUBSETS=(dh_midpoint td_ego_dir td_ego_dir_arrow td_ego_side td_ego_side_arrow td_midpoint td_path td_path_arrow)
for SUBSET in "${SUBSETS[@]}"; do
    for SIDEVIEW in "" "_sideview"; do
        NAME="pt2p_${SUBSET}${SIDEVIEW}"
        # Find latest log for this job
        LOG=$(ls -t ${LOG_DIR}/${NAME}_*.out 2>/dev/null | head -1)
        if [ -z "$LOG" ]; then
            echo "${NAME}: NO LOG"
            continue
        fi
        JOB_ID=$(echo "$LOG" | grep -oP '\d+(?=\.out)')
        # Check if job is still in queue
        if squeue -u linjli --format="%i" -h 2>/dev/null | grep -q "^${JOB_ID}$"; then
            # Get progress from log
            PROGRESS=$(sed -n '$p' "$LOG" | tr '\r' '\n' | grep "Infer" | tail -1 | grep -oP '\d+%' | tail -1)
            echo "${NAME} (${JOB_ID}): RUNNING ${PROGRESS:-loading...}"
        else
            # Job finished - check for errors
            if grep -q "SIGBUS\|CANCELLED\|FAILED" "$LOG" 2>/dev/null; then
                echo "${NAME} (${JOB_ID}): FAILED"
            elif find ${OUTPUT_DIR} -name "*AI2ThorPT2P_${SUBSET}${SIDEVIEW}*_acc.csv" 2>/dev/null | grep -q .; then
                echo "${NAME} (${JOB_ID}): COMPLETED"
            else
                PROGRESS=$(sed -n '$p' "$LOG" | tr '\r' '\n' | grep "Infer" | tail -1 | grep -oP '\d+%' | tail -1)
                echo "${NAME} (${JOB_ID}): FINISHED (${PROGRESS:-unknown}%)"
            fi
        fi
    done
done
