#!/bin/bash
LOG_DIR=/gpfs/home/linjli/source/SpatialReasoning_Eval/slurm_logs_pt2p

# Define job IDs and names
declare -A JOBS
JOBS[58429]="dh_midpoint (no sv)"
JOBS[58430]="dh_midpoint (sideview)"
JOBS[58431]="td_midpoint (no sv)"
JOBS[58432]="td_midpoint (sideview)"
JOBS[58433]="td_ego_dir (no sv)"
JOBS[58434]="td_ego_dir (sideview)"
JOBS[58435]="td_ego_dir_arrow (no sv)"
JOBS[58436]="td_ego_dir_arrow (sideview)"
JOBS[58437]="td_ego_side_arrow (no sv)"
JOBS[58438]="td_ego_side_arrow (sideview)"
JOBS[58439]="td_ego_side (no sv)"
JOBS[58440]="td_ego_side (sideview)"
JOBS[58441]="td_path (no sv)"
JOBS[58442]="td_path (sideview)"
JOBS[58443]="td_path_arrow (no sv)"
JOBS[58444]="td_path_arrow (sideview)"

# Job names to log file prefix mapping
declare -A JOB_LOG_PREFIX
JOB_LOG_PREFIX[58429]="pt2p_dh_midpoint_58429"
JOB_LOG_PREFIX[58430]="pt2p_dh_midpoint_sideview_58430"
JOB_LOG_PREFIX[58431]="pt2p_td_midpoint_58431"
JOB_LOG_PREFIX[58432]="pt2p_td_midpoint_sideview_58432"
JOB_LOG_PREFIX[58433]="pt2p_td_ego_dir_58433"
JOB_LOG_PREFIX[58434]="pt2p_td_ego_dir_sideview_58434"
JOB_LOG_PREFIX[58435]="pt2p_td_ego_dir_arrow_58435"
JOB_LOG_PREFIX[58436]="pt2p_td_ego_dir_arrow_sideview_58436"
JOB_LOG_PREFIX[58437]="pt2p_td_ego_side_arrow_58437"
JOB_LOG_PREFIX[58438]="pt2p_td_ego_side_arrow_sideview_58438"
JOB_LOG_PREFIX[58439]="pt2p_td_ego_side_58439"
JOB_LOG_PREFIX[58440]="pt2p_td_ego_side_sideview_58440"
JOB_LOG_PREFIX[58441]="pt2p_td_path_58441"
JOB_LOG_PREFIX[58442]="pt2p_td_path_sideview_58442"
JOB_LOG_PREFIX[58443]="pt2p_td_path_arrow_58443"
JOB_LOG_PREFIX[58444]="pt2p_td_path_arrow_sideview_58444"

MAX_CHECKS=60
CHECK_INTERVAL=300  # 5 minutes

for ((i=1; i<=MAX_CHECKS; i++)); do
    echo "=== Check $i at $(date) ==="
    
    completed=0
    failed=0
    running=0
    
    for jid in $(echo "${!JOBS[@]}" | tr ' ' '\n' | sort -n); do
        name="${JOBS[$jid]}"
        logfile="${LOG_DIR}/${JOB_LOG_PREFIX[$jid]}.out"
        
        # Check if job is still running
        state=$(sacct -j $jid --format=State -n 2>/dev/null | head -1 | tr -d ' ')
        
        if [ -f "$logfile" ]; then
            acc=$(grep -P "^Accuracy" "$logfile" 2>/dev/null | head -1 | awk '{print $2}')
            count=$(grep -P "^Count" "$logfile" 2>/dev/null | head -1 | awk '{print $2}')
        fi
        
        if [ -n "$acc" ]; then
            echo "  DONE  $jid $name: Accuracy=$acc Count=$count"
            ((completed++))
        elif [ "$state" = "RUNNING" ]; then
            prog=$(grep -oP "Rank 0/2:\s+\d+%" "$logfile" 2>/dev/null | tail -1)
            echo "  RUN   $jid $name: $prog"
            ((running++))
        elif [ "$state" = "COMPLETED" ] || [ "$state" = "FAILED" ] || [ "$state" = "TIMEOUT" ] || [ "$state" = "CANCELLED" ]; then
            echo "  $state $jid $name (no accuracy found)"
            ((failed++))
        else
            echo "  ???   $jid $name state=$state"
            ((running++))
        fi
    done
    
    echo "  Summary: $completed done, $running running, $failed failed/unknown"
    
    if [ $((completed + failed)) -eq 16 ]; then
        echo ""
        echo "=== ALL JOBS FINISHED ==="
        echo ""
        echo "| Subset | No Sideview (Acc%) | Sideview (Acc%) |"
        echo "|--------|-------------------|-----------------|"
        
        for subset in dh_midpoint td_midpoint td_ego_dir td_ego_dir_arrow td_ego_side td_ego_side_arrow td_path td_path_arrow; do
            nosv_log=$(ls ${LOG_DIR}/pt2p_${subset}_584*.out 2>/dev/null | grep -v sideview | head -1)
            sv_log=$(ls ${LOG_DIR}/pt2p_${subset}_sideview_584*.out 2>/dev/null | head -1)
            
            nosv_acc=$(grep -P "^Accuracy" "$nosv_log" 2>/dev/null | head -1 | awk '{print $2}')
            sv_acc=$(grep -P "^Accuracy" "$sv_log" 2>/dev/null | head -1 | awk '{print $2}')
            
            [ -z "$nosv_acc" ] && nosv_acc="N/A"
            [ -z "$sv_acc" ] && sv_acc="N/A"
            
            echo "| $subset | $nosv_acc | $sv_acc |"
        done
        
        exit 0
    fi
    
    echo "  Waiting ${CHECK_INTERVAL}s..."
    echo ""
    sleep $CHECK_INTERVAL
done

echo "Monitoring timed out after $MAX_CHECKS checks."
