# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

Leaf_FOLDER="Leaf"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/${EXP_FOLDER}/train"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/${EXP_FOLDER}/vis"
DATASET="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/tfrecord/cv_1/tfrecord_cv1_split1"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/${EXP_FOLDER}/results_train.txt"

python3 "${WORK_DIR}"/eval_leaf.py \
--logtostderr \
--vis_split="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--vis_crop_size=512,512 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--vis_logdir="${VIS_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset="leaf" \
--dataset_dir="${DATASET}" \
--max_number_of_iterations=1
