#!/bin/bash
#PBS -N dru_bs2_fnd_kitti_source_same_hyperparam
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=72:0:00
#PBS -o pbs_logs/fnd_kitti_bs2_source_same_hyperparam_outputs.log
#PBS -e pbs_logs/fnd_kitti_bs2_source_same_hyperparam_errors.log

cd 
source ~/scratch/setExport_sarvesh.sh
cd $PBS_O_WORKDIR
source /home/s_shashi/scratch/anaconda3/etc/profile.d/conda.sh
ls

conda activate dru_double

N_GPUS=1
BATCH_SIZE=2
DATA_ROOT=/home/s_shashi/scratch/Negroni_Dataset/Coco_Data/Natural
OUTPUT_DIR=./outputs/def-detr-base/kitti2city/source_only_fnd_same_hyperparam

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26503 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone focalnet_L_384_22k \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset kitti \
--target_dataset cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 0.0001 \
--lr_backbone 1e-05 \
--lr_linear_proj 2e-5 \
--epoch 80 \
--epoch_lr_drop 11 \
--mode single_domain \
--output_dir ${OUTPUT_DIR} \
--detector fnd \
--config_file DINO_4scale_focalnet_large_fl3.py \
--weight_decay 0.0001 \
--clip_max_norm 0.1 \
--hidden_dim 256 \
--num_queries 900 \
--num_feature_levels 4 \
--random_seed 42 \
--resume /home/s_shashi/scratch/Repos/FMA_PNP/DRU/pretrained_backbone/dino_focal_large_3level_4scale_36ep.pth \
--finetune_ignore label_enc.weight class_embed |& tee tee_logs/fnd_bs2_source_kitti_same_hyperparam.txt