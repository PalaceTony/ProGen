#!/bin/bash

source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate s4Env
WEBHOOK_URL="https://hooks.slack.com/services/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"

export MODEL_NAME="random_run_pems07"  
export DATASET="PEMS08"
export DEVICE_FULL="1a80011-0"
export SAMPLING_FIRST_BATCH=11 
export RANDOM_SAMPLING=True
export BEST_MODEL_PATH="outputs/score_train/2024-06-10/20-48-53/trial_56_0613_11_49_15/best_model.pth"
export LOG_TO_DEBUG_FILE=False

if [ "$SAMPLING_FIRST_BATCH" = "all" ]; then
    export SAMPLING_FIRST_BATCH_VALUE=20000000000
elif [ "$SAMPLING_FIRST_BATCH" = "random" ]; then
    export SAMPLING_FIRST_BATCH_VALUE=11
else
    export SAMPLING_FIRST_BATCH_VALUE="$SAMPLING_FIRST_BATCH"
fi

export NCCL_SOCKET_IFNAME=eth2
export NCCL_TIMEOUT=3600

# Array of sample sizes
samples=(30)

for sample in "${samples[@]}"; do
    for c_value in $(seq -0.3 0.1 1.0); do
        export NOW=$(date -d '+8 hours' +"%m-%d_%H-%M-%S")
        export DATE=$(date -d '+8 hours' +"%Y-%m-%d")
        export TIME=$(date -d '+8 hours' +"%H-%M-%S")
        export UNIQUE_RUN_ID=$(date -d '+8 hours' +"%H-%M-%S")
        export OUTPUT_DIR="outputs/sampling/${DATE}/${TIME}_c_${c_value}_samples_${sample}"
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
        "excel_notes='${DEVICE_FULL}. ${MODEL_NAME}. NewSDE used for sampling only. no diffusion. Sampling. thesis. ${DATASET}. ${sample} samples. ${SAMPLING_FIRST_BATCH_VALUE} batches. Neighbors sum c=${c_value}'" \
        "n_samples=${sample}" \
        "random_sampling=${RANDOM_SAMPLING}" \
        "best_model_path=${BEST_MODEL_DIR}/" \
        "dataset=${DATASET}" \
        "dataset.batch_size=${SAMPLING_FIRST_BATCH}" \
        "dataset.dif_model.model.nonlinearity=relu" \
        "dataset.neighbors_sum_c=${c_value}" \
        "dataset.neighbors_sum_c2=0" \
        "dataset.dif_model.model.nf=32" \
        "dataset.dif_model.stlayer.hidden_size=95" \
        "dataset.dif_model.model.pos_emb=16" \
        "dataset.dif_model.model.num_res_blocks=4" \
        "dataset.dif_model.model.ch_mult=[1, 2]" \
        "dataset.column_wise=False" \
        ${SAMPLING_FIRST_BATCH_VALUE:+ "sampling_first_batch=${SAMPLING_FIRST_BATCH_VALUE}"}
    done
done

if [ $? -eq 0 ]; then
    MESSAGE="Script 0728_pems08_save_each_iteration_clean_newSTlayer_previous_norm_spatialSDE_corrector_optuna.sh completed successfully!"
    curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$MESSAGE"'"}' $WEBHOOK_URL
else
    MESSAGE="Script 0728_pems08_save_each_iteration_clean_newSTlayer_previous_norm_spatialSDE_corrector_optuna.sh failed!"
    curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$MESSAGE"'"}' $WEBHOOK_URL
    exit 1
fi

# Change directory and run the final script
cd ../spatiotemporal_diffusion
./test1gpu.sh