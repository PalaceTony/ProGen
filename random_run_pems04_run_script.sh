#!/bin/bash
WEBHOOK_URL="https://hooks.slack.com/services/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"


source /hpc2hdd/home/mgong081/anaconda3/etc/profile.d/conda.sh
conda activate s4Env

export MODEL_NAME="random_run_pems03_04_08"  
export DATASET="PEMS04"
export DEVICE_FULL="1a800-0"
export SAMPLING_FIRST_BATCH=11
export RANDOM_SAMPLING=True
export BEST_MODEL_PATH="outputs/score_train/2024-06-10/20-48-39/trial_46_0613_07_17_16/best_model.pth"
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

# Array of seeds
seeds=(42)

for seed in "${seeds[@]}"; do
    export NOW=$(date -d '+8 hours' +"%m-%d_%H-%M-%S")
    export DATE=$(date -d '+8 hours' +"%Y-%m-%d")
    export TIME=$(date -d '+8 hours' +"%H-%M-%S")
    export UNIQUE_RUN_ID=$(date -d '+8 hours' +"%H-%M-%S")
    export OUTPUT_DIR="outputs/sampling/${DATE}/${TIME}_seed_${seed}"
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
    "excel_notes='${DEVICE_FULL}. ${MODEL_NAME}. NewSDE used for sampling only. no diffusion. Sampling. thesis. ${DATASET}. ${SAMPLES} samples. ${SAMPLING_FIRST_BATCH_VALUE} batches. Seed ${seed}'" \
    "n_samples=${SAMPLES}" \
    "random_sampling=${RANDOM_SAMPLING}" \
    "best_model_path=${BEST_MODEL_DIR}/" \
    "dataset=${DATASET}" \
    "dataset.batch_size=${SAMPLING_FIRST_BATCH}" \
    "dataset.dif_model.model.nonlinearity=relu" \
    ${SAMPLING_FIRST_BATCH_VALUE:+ "sampling_first_batch=${SAMPLING_FIRST_BATCH_VALUE}"} \
    "dataset.neighbors_sum_c=0.6125045418739319" \
    "dataset.neighbors_sum_c2=0" \
    "dataset.dif_model.model.nf=256" \
    "dataset.dif_model.stlayer.hidden_size=242" \
    "dataset.dif_model.model.pos_emb=128" \
    "dataset.dif_model.model.num_res_blocks=3" \
    "dataset.dif_model.model.ch_mult=[1, 2]" \
    "dataset.column_wise=False" \
    "seed=${seed}" \
    # "sampling_num_batches_to_process=1" \
    # "random_sampling=True" \
    # "batch_size=64" \
    # "dataset.dif_model.stlayer.hidden_size=64" \
    # "dif_model.stlayer.hidden_size=64" \
    # "wandb_mode=dryrun" \
    # "dataset.data_path=data/raw/pems08.npz" \
    # "dataset.adj=data/raw/PEMS08.csv" \
    # "epochs=1" \
    # "num_batches_to_process=1" \
    # "dif_model.model.num_scales=10" \
    # "best_model_path='/hpc2hdd/home/mgong081/Projects/spatiotemporal_diffusion/outputs/score_train/2024-05-25/00-41-16/00-41-16/'" \
    # "dif_model.model.num_scales=2" \
    # "sampling_first_batch=1" \
    # "dif_model.model.beta_max=24" \
    # "lr_init=0.004031259020238426" \
    # "num_batches_to_process=1" \
    # "epochs=1" \
    # "dif_model.model.num_scales=10" \
    # "hyperopt=True" \
    # "hyperopt_trial=50" \
    # "dif_model.model.nf=128" \
    # "hyperopt=True" \
    # "hyperopt_trial=100" \
    # "train_sampling=True" \
    # "lr_init=0.002" \
    # "excel_notes='6a40-5. SAMPLING ONLY. trial_13_0602_23_29_23. simple contact. 1 batch. random 8. 50 samples.'" \
done

if [ $? -eq 0 ]; then
    MESSAGE="Script 0728_pems04_seed_clean_newSTlayer_previous_norm_spatialSDE_corrector_optuna.sh completed successfully!"
    curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$MESSAGE"'"}' $WEBHOOK_URL
else
    MESSAGE="Script 0728_pems04_seed_clean_newSTlayer_previous_norm_spatialSDE_corrector_optuna.sh failed!"
    curl -X POST -H 'Content-type: application/json' --data '{"text":"'"$MESSAGE"'"}' $WEBHOOK_URL
    exit 1
fi