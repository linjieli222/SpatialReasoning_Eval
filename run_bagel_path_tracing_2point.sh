#!/bin/bash
# Evaluate BAGEL-7B-MoT on AI2Thor Path Tracing 2-Point dataset
# 8 subsets x 2 (with/without sideview) = 16 evaluations

set -e

WORK_DIR=./outputs_bagel_pt2p

echo "========================================"
echo "BAGEL-MoT Path Tracing 2-Point Evaluation"
echo "========================================"

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

TOTAL=$((${#SUBSETS[@]} * 2))
COUNT=0

for SUBSET in "${SUBSETS[@]}"; do
    # Without sideview (topdown only)
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] AI2ThorPT2P_${SUBSET} (topdown only)"
    echo "----------------------------------------"
    python run.py --model bagel_mot --data "AI2ThorPT2P_${SUBSET}" --work-dir "$WORK_DIR"

    # With sideview (topdown + sideview)
    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] AI2ThorPT2P_${SUBSET}_sideview (topdown + sideview)"
    echo "----------------------------------------"
    python run.py --model bagel_mot --data "AI2ThorPT2P_${SUBSET}_sideview" --work-dir "$WORK_DIR"
done

echo ""
echo "========================================"
echo "All $TOTAL evaluations completed!"
echo "Results saved in: $WORK_DIR"
echo "========================================"
