import time
import datetime
import json

import copy

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator

from models.criterion import post_process, get_pseudo_labels, get_topk_outputs, SetCriterion
from utils.distributed_utils import is_main_process
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List

from datasets.masking import Masking
from scipy.optimize import linear_sum_assignment
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils import selective_reinitialize

from groundingdino.util.inference import predict
import math
import os
from PIL import ImageDraw, ImageFont
from PIL import Image


class ModelNotFoundError(Exception):
    pass


class CosineWarmup:
    def __init__(self, T0, T, device="cpu"):
        """
        Cosine warmup function for scaling L_exp.

        Args:
        - T0 (int): Delay before L_exp starts contributing (e.g., 30 epochs).
        - T (int): Total number of epochs.
        - device (str): Device to store the lambda tensor.
        """
        self.T0 = T0
        self.T = T
        self.device = device

    def get_lambda(self, epoch):
        """
        Compute the weight for L_exp at a given epoch.

        Args:
        - epoch (int): Current training epoch.

        Returns:
        - lambda_exp (float): Scaling factor for L_exp.
        """
        if epoch < self.T0:
            return 0.0  # L_exp is completely turned off
        else:
            progress = (epoch - self.T0) / (self.T - self.T0)
            return 0.5 * (1 - math.cos(math.pi * progress))

def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True,
                             model_name: str = 'def_detr'):
    """
    Train the standard detection model, using only labelled training set source.
    """
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        if model_name == 'fnd':
            out = model(images, masks, annotations)
        else:
            out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching_standard(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks, dru_teacher=True)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'], teacher_out['boxes_all'], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks, pseudo_labels)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_teaching_mask(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks, dru_teacher=True)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'], teacher_out['boxes_all'], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks, pseudo_labels)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks, pseudo_labels)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        loss = target_loss + coef_masked_img * masked_target_loss

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks, dru_teacher=True)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            stu_buffer_cost.append(var_total)

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks, dru_teacher=True)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'], init_student_out['boxes_all'],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights=use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights=use_pseudo_label_weights)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:
                    res_dict['stu_ori'].append(stu_buffer_cost)
                    res_dict['stu_now'].append(all_score)
                    res_dict['update_iter'].append(len(stu_buffer_cost))

                    df = pd.DataFrame(res_dict)
                    df.to_csv('dynamic_update.csv')

                    with torch.no_grad():
                        state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                        for key, value in state_dict.items():
                            state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                        teacher_model.load_state_dict(state_dict)

                    # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
            else:
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_teaching_mask_double(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  expert_model,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False,
                                  text_prompt: str = "person . car . train . rider . truck . motorcycle . bicycle . bus .",
                                  box_threshold: float = 0.35,
                                  text_threshold: float = 0.25,
                                  label_classes: list = ["person", "car", "train", "rider", "truck", "motorcycle", "bicycle", "bus"],
                                  expert_detector = None,
                                  expert_model_type: str = "groundingdino"):
    """
    Train the student model with the teacher model, and expert model, using only unlabeled training set target (plus masked target image)
    """
    # T0 = 10 #Start intrducing L_exp from this epoch
    # T = 60 #Total epochs of training
    # T0 = 0 #Start intrducing L_exp from this epoch
    # T = 14 #Total epochs of training
    # warmup_scheduler = CosineWarmup(T0, T)
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)
    image_no = 0
    for iter in range(total_iters):
        # target_images, target_masks, _ = target_fetcher.next()
        # if target_images is not None:
        #     target_teacher_images, target_student_images = target_images[0], target_images[1]
        # continue
        expert_labels = []

        # Target teacher forward
        with torch.no_grad():
            # Resize target_teacher_images, target_student_images for weak_aug_meds
            teacher_out = teacher_model(target_teacher_images, target_masks, dru_teacher=True)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'], teacher_out['boxes_all'], thresholds)                

            #With Grouding DINO supervision
            for img in target_teacher_images:
                boxes, logits, phrases = predict(
                    model=expert_model,
                    image=img,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    remove_combined=True
                )
                # Convert phrases to their corresponding indices in label_classes (index + 1)
                indices = [label_classes.index(phrase) + 1 for phrase in phrases if phrase in label_classes]
                logits = logits.to(device)
                boxes = boxes.to(device)
                # Convert indices list to tensor
                tensor_indices = torch.tensor(indices, device=device)
                expert_labels.append({'boxes' : torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).to("cuda", non_blocking=True),
                'scores': torch.tensor(logits, dtype=torch.double).to("cuda", non_blocking=True),
                'labels': torch.tensor(tensor_indices, dtype=torch.int64).to("cuda", non_blocking=True)
                })
                        
        # Target student forward
        target_student_out1, target_student_out2 = student_model(target_student_images, target_masks, pseudo_labels)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out1, pseudo_labels)
        try:
            target_loss_expert, target_loss_dict_expert = criterion_pseudo(target_student_out2, expert_labels)
        except Exception as e:
            print("ERROR IN CRITERION!!!!")
            print(e)
            print("STUDENT OUT:")
            print(target_student_out2)
            print("*"*20)
            print("GDINO OUT:")
            print(expert_labels)
            raise RuntimeError
            
        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out1, masked_target_student_out2 = student_model(masked_target_images, target_masks, pseudo_labels)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out1, pseudo_labels)
        masked_target_loss_expert, masked_target_loss_dict_expert = criterion_pseudo(masked_target_student_out2, expert_labels)

        # Final loss
        if epoch < 25 :
            lambda_exp = 1
        else:
            lambda_exp = 1
        lambda_csod = 1
        lambda_hist = 1
        if iter == 0:
            # print(f"Annealing as follows (Lambda_exp) for epoch {epoch}: {lambda_exp}")
            print("Lambda CSOD: ", lambda_csod)
            print("Lambda HIST: ", lambda_hist)
            print("Lambda EXP: ", lambda_exp)
        target_loss_combined = lambda_csod * target_loss + lambda_exp * target_loss_expert #Annealing for the Expert Loss
        masked_target_loss_combined = lambda_csod * masked_target_loss + lambda_exp * masked_target_loss_expert #Annealing for the Expert Loss
        loss = target_loss_combined + coef_masked_img * masked_target_loss_combined

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out1, student_out2 = student_model(target_teacher_images, target_masks, dru_teacher=True)
            # variance logit
            student_out_var1 = student_out1['logit_all'].var(dim=0)
            var_total1 = student_out_var1.mean().item()
            student_out_var2 = student_out2['logit_all'].var(dim=0)
            var_total2 = student_out_var2.mean().item()
            var_total = (var_total1 + var_total2) / 2
            stu_buffer_cost.append(var_total)
            
            # Store batch data to buffer            
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out1, init_student_out2 = init_student_model(target_teacher_images, target_masks, dru_teacher=True)
                    pseudo_labels_init_student1 = get_pseudo_labels(init_student_out1['logit_all'], init_student_out1['boxes_all'],
                                                              thresholds)
                    pseudo_labels_init_student2 = get_pseudo_labels(init_student_out2['logit_all'], init_student_out2['boxes_all'],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss1, init_student_loss_dict1 = criterion_pseudo_weak(target_student_out1,
                                                                                    pseudo_labels_init_student1, use_pseudo_label_weights=use_pseudo_label_weights)
                init_student_loss2, init_student_loss_dict2 = criterion_pseudo_weak(target_student_out2,
                                                                                    pseudo_labels_init_student2, use_pseudo_label_weights=use_pseudo_label_weights)
                masked_init_student_loss1, masked_init_student_loss_dict1 = criterion_pseudo_weak(masked_target_student_out1,
                                                                                                  pseudo_labels_init_student1, use_pseudo_label_weights=use_pseudo_label_weights)
                masked_init_student_loss2, masked_init_student_loss_dict2 = criterion_pseudo_weak(masked_target_student_out2,
                                                                                                  pseudo_labels_init_student2, use_pseudo_label_weights=use_pseudo_label_weights)
                init_student_loss = init_student_loss1 + init_student_loss2
                masked_init_student_loss = masked_init_student_loss1 + masked_init_student_loss2
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += lambda_hist * loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu_double(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:
                    res_dict['stu_ori'].append(stu_buffer_cost)
                    res_dict['stu_now'].append(all_score)
                    res_dict['update_iter'].append(len(stu_buffer_cost))

                    df = pd.DataFrame(res_dict)
                    df.to_csv('dynamic_update.csv')

                    with torch.no_grad():
                        state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                        for key, value in state_dict.items():
                            state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                        teacher_model.load_state_dict(state_dict)

                    # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
            else:
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict
    

@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False,
             postprocessors: dict = None):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        # dataset_annotations = [[] for _ in range(len(coco_data['images']))]
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        try:
            out, out2 = model(images, masks)
        except:
            out = model(images, masks)
        logit_all, boxes_all = out['logit_all'], out['boxes_all']
        # Get pseudo labels
        if not output_result_labels:
            results = get_pseudo_labels(logit_all, boxes_all, [0.4 for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    # dataset_annotations[image_id].append(pseudo_anno)
                    dataset_annotations[image_id].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        if postprocessors is None:
            results = post_process(logit_all[-1], boxes_all[-1], orig_image_sizes, 100)
        else:
            results = postprocessors['bbox'](out, orig_image_sizes)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        # for image_anno in dataset_annotations:
        for image_anno in dataset_annotations.values():
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)


def eval_stu(student_model: torch.nn.Module,
             stu_buffer_img: List[torch.Tensor],
             stu_buffer_mask: List[torch.Tensor]):
    """
    Evaluate student model with variance of logit
    """
    student_model.eval()
    all_score = []
    with torch.no_grad():
        for i in range(len(stu_buffer_img)):
            # student_out['logit_all']: [num_decoder_layers, batch size, num_queries, num_classes]
            student_out = student_model(stu_buffer_img[i], stu_buffer_mask[i])

            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            all_score.append(var_total)

    return all_score


def eval_stu_double(student_model: torch.nn.Module,
             stu_buffer_img: List[torch.Tensor],
             stu_buffer_mask: List[torch.Tensor]):
    """
    Evaluate student model with variance of logit
    """
    student_model.eval()
    all_score = []
    with torch.no_grad():
        for i in range(len(stu_buffer_img)):
            # student_out['logit_all']: [num_decoder_layers, batch size, num_queries, num_classes]
            student_out1,  student_out2 = student_model(stu_buffer_img[i], stu_buffer_mask[i])
            student_out_var1 = student_out1['logit_all'].var(dim=0)
            student_out_var2 = student_out2['logit_all'].var(dim=0)

            var_total1 = student_out_var1.mean().item()
            var_total2 = student_out_var2.mean().item()

            var_total = (var_total1 + var_total2) / 2
            all_score.append(var_total)

    return all_score


def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)
    

@torch.no_grad()
def visualize(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False,
             postprocessors: dict = None,):
    start_time = time.time()
    model.eval()
    criterion.eval()

    colors = {
                "2": (0, 0, 255),    # Blue
                "1": (0, 255, 0),    # Green
                "9": (255, 0, 0),    # Red
                "5": (255, 255, 0),  # Yellow
                "4": (255, 0, 255),  # Magenta
                "7": (0, 255, 255),  # Cyan
                "6": (128, 0, 128),  # Purple
                "8": (255, 165, 0),  # Orange
                "3": (128, 128, 128) # Gray
    }
    # font = ImageFont.load_default()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        # dataset_annotations = [[] for _ in range(len(coco_data['images']))]
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')

    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        # Forward
        out = model(images, masks)
        # logit_all, boxes_all = out['logit_all'], out['boxes_all']
        outputs = postprocessors['bbox'](out, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        image_id = annotations[0]['image_id'].item()
        file_name = next(
                            img["file_name"]
                            for img in coco_data["images"]
                            if img["id"] == image_id
                        )
        image_ = Image.open(os.path.join(data_loader_val.dataset.image_root, file_name)).convert("RGB")
        draw = ImageDraw.Draw(image_, "RGBA")
        orig_image_size = annotations[0]['orig_size']
        img_h, img_w = orig_image_size.unbind(0)
        # outputs = post_process(logit_all[-1], boxes_all[-1], orig_image_size, 100)
        # img_w, img_h = image_.size
        scale_fct = torch.stack([torch.tensor(img_w, device="cuda"), torch.tensor(img_h, device="cuda"), torch.tensor(img_w, device="cuda"), torch.tensor(img_h, device="cuda")])

        for idx, output in enumerate(outputs):
            converted_boxes = outputs['boxes'] * scale_fct
            converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
            for label, box, score in zip(outputs['labels'].detach().cpu().numpy().tolist(), converted_boxes, outputs['scores'].detach().cpu().numpy().tolist()):
                if score < 0.3:
                    continue
                x1, y1, x2, y2 = box  
                draw.rectangle([x1, y1, x2, y2], outline=colors[str(label)], width=6)
                # ----- Add text above the box -----
                # if label == 2:
                #     print(label)
                #     continue
                text = coco_data["categories"][label - 1]["name"]
                print(text)

                # Compute text size (Pillow <10 style)
                text_width, text_height = draw.textsize(text, font=font)

                # Text position (slightly above y1)
                text_x = x1
                text_y = y1 - text_height - 4  # 4px padding

                # # Optional background for readability
                # draw.rectangle(
                #     [text_x, text_y, text_x + text_width, text_y + text_height],
                #     fill=color
                # )

                # Draw the text
                draw.text((text_x, text_y), text, fill=colors[str(label)], font=font)

        batch = image_id // 16
        num = image_id % 16

        if batch > 6:
            break

        image_.save(f"city2bdd/{batch}_{num}_fnd_double.png")