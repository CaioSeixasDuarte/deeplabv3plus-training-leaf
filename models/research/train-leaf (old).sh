# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
Leaf_FOLDER="Leaf"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/${EXP_FOLDER}/train"
DATASET="${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${Leaf_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"

python3 "${WORK_DIR}"/train.py \
  --logtostderr \
  --base_learning_rate=.005 \
  --training_number_of_steps=20000 \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=4 \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --fine_tune_batch_norm=False \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"
