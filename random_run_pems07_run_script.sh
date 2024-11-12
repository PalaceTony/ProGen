#!/bin/bash

source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate s4Env
WEBHOOK_URL="https://hooks.slack.com/services/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"

export MODEL_NAME="0728_clean_newSTlayer_greatest4_norm_spatialSDE_corrector_optuna_pems07"  
export DATASET="PEMS07"
export DEVICE_FULL="1a80011-0"
export SAMPLING_FIRST_BATCH=11 
export RANDOM_SAMPLING=True
export BEST_MODEL_PATH="outputs/score_train/2024-06-14/02-14-55/trial_16_0616_08_03_10/best_model.pth"
export LOG_TO_DEBUG_FILE=False
export SAMPLES=30

if [ "$SAMPLING_FIRST_BATCH" = "all" ]; then
    export SAMPLING_FIRST_BATCH_VALUE=20000000000
elif [ "$SAMPLING_FIRST_BATCH" = "random" ]; then
    export SAMPLING_FIRST_BATCH_VALUE=11
else
    export SAMPLING_FIRST_BATCH_VALUE="$SAMPLING_FIRST_BATCH"
fi

export NCCL_SOCKET_IFNAME=eth2
export NCCL_TIMEOUT=3600

# Array of num_scales values
num_scales_values=(1000)

# Array of neighbors_sum_c values
neighbors_sum_c_values=(0 0.5125045418739319)

for num_scales in "${num_scales_values[@]}"; do
    for neighbors_sum_c in "${neighbors_sum_c_values[@]}"; do
        export NOW=$(date -d '+8 hours' +"%m-%d_%H-%M-%S")
        export DATE=$(date -d '+8 hours' +"%Y-%m-%d")
        export TIME=$(date -d '+8 hours' +"%H-%M-%S")
        export UNIQUE_RUN_ID=$(date -d '+8 hours' +"%H-%M-%S")
        export OUTPUT_DIR="outputs/sampling/${DATE}/${TIME}_num_scales_${num_scales}_neighbors_sum_c_${neighbors_sum_c}"
        export DEVICE="cuda:${DEVICE_FULL: -1}"
        export BEST_MODEL_DIR=$(dirname "${BEST_MODEL_PATH}")

        # Save the current script to a temporary file
        SCRIPT_FILE=$(mktemp)
        cat "$0" > "$SCRIPT_FILE"
        # Ensure the output directory exists
        mkdir -p ${OUTPUT_DIR}
        # Copy the temporary script file to the output directory
        cp "$SCRIPT_FILE" "${OUTPUT_DIR}/run_script.sh"
        # Clean up the temporary script file
        rm "$SCRIPT_FILE"

        python src/model/${MODEL_NAME}/v011_run_sde.py \
        "hyperopt=False" \
        "hyperopt_trial=300" \
        "log_to_debug_file=${LOG_TO_DEBUG_FILE}" \
        "use_distributed=False" \
        "test_only=True" \
        "log_dir=${OUTPUT_DIR}" \
        "run_name=${NOW}" \
        "wandb_project_name='${MODEL_NAME}_${DATASET}'" \
        "wandb_mode='dryrun'" \
        "device=${DEVICE}" \
        "excel_notes='${DEVICE_FULL}. ${MODEL_NAME}. ${neighbors_sum_c}. Sampling. ${DATASET}. ${SAMPLING_FIRST_BATCH_VALUE} batches.'" \
        "n_samples=${SAMPLES}" \
        "random_sampling=${RANDOM_SAMPLING}" \
        "best_model_path=${BEST_MODEL_DIR}/" \
        "dataset=${DATASET}" \
        "dataset.batch_size=${SAMPLING_FIRST_BATCH}" \
        "dataset.dif_model.model.nonlinearity=relu" \
        "dataset.dif_model.model.num_scales=${num_scales}" \
        ${SAMPLING_FIRST_BATCH_VALUE:+ "sampling_first_batch=${SAMPLING_FIRST_BATCH_VALUE}"} \
        "dataset.neighbors_sum_c=${neighbors_sum_c}" \
        "dataset.neighbors_sum_c2=0" \
        "dataset.dif_model.model.nf=32" \
        "dataset.dif_model.stlayer.hidden_size=97" \
        "dataset.dif_model.model.pos_emb=32" \
        "dataset.dif_model.model.num_res_blocks=3" \
        "dataset.dif_model.model.ch_mult=[1, 2]" \
        "dataset.column_wise=False" \
        "seed=42"

        if [ $? -eq 0 ]; then
            MESSAGE="Script completed successfully for neighbors_sum_c=${neighbors_sum_c}!"
            curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$MESSAGE"'"}' $WEBHOOK_URL
        else
            MESSAGE="Script failed for neighbors_sum_c=${neighbors_sum_c}!"
            curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$MESSAGE"'"}' $WEBHOOK_URL
            exit 1
        fi
    done
done

# Change directory and run the final script
cd ../spatiotemporal_diffusion
./test1gpu.sh
