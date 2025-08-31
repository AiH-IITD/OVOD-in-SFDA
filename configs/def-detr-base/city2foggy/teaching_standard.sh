#!/bin/bash
#PBS -N dru_bs2_fnd_cs_mt
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=96:0:00
#PBS -o pbs_logs/fnd_cs_bs2_mt.log
#PBS -e pbs_logs/fnd_cs_bs2_mt.log

cd 
source ~/scratch/setExport_sarvesh.sh
cd $PBS_O_WORKDIR
source /home/s_shashi/scratch/anaconda3/etc/profile.d/conda.sh
ls

conda activate dru_double

N_GPUS=1
BATCH_SIZE=2
DATA_ROOT=/home/s_shashi/scratch/Negroni_Dataset/Coco_Data/Natural
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/teaching_standard

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26505 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone focalnet_L_384_22k \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--alpha_ema 0.999 \
--epoch 30 \
--epoch_lr_drop 80 \
--mode teaching_standard \
--threshold 0.3 \
--fix_update_iter 1 \
--output_dir ${OUTPUT_DIR} \
--detector fnd \
--config_file DINO_4scale_focalnet_large_fl3.py \
--weight_decay 0.0001 \
--clip_max_norm 0.1 \
--hidden_dim 256 \
--num_queries 900 \
--num_feature_levels 4 \
--random_seed 42 \
--resume /home/s_shashi/scratch/Repos/FMA_PNP/DRU/outputs/def-detr-base/city2foggy/source_only_fnd_same_hyperparam/model_best.pth |& tee tee_logs/fnd_bs2_mt.txt