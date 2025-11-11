#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss функции для Knowledge Distillation
DistillKL + CAMKD (Cross-teacher Attentive Multi-teacher KD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistillKL(nn.Module):
    """
    KL divergence loss для Knowledge Distillation
    Использует temperature scaling
    """
    def __init__(self, T=4.0):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        """
        Args:
            y_s: student logits (batch_size, num_classes)
            y_t: teacher logits (batch_size, num_classes)

        Returns:
            KL divergence loss
        """
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


class CAMKD(nn.Module):
    """
    Cross-teacher Attentive Multi-teacher Knowledge Distillation

    Особенности:
    - Адаптивные веса учителей: w_i = (1 - softmax(loss_t)) / (M-1)
    - Feature distillation (MSE loss)
    - Логирование весов
    """
    def __init__(self):
        super(CAMKD, self).__init__()

    def forward(self, student_feat_list, teacher_feat_list, teacher_logits_list, labels):
        """
        Args:
            student_feat_list: list of student features (после ConvReg)
            teacher_feat_list: list of teacher features
            teacher_logits_list: list of teacher logits
            labels: ground truth labels

        Returns:
            loss: weighted feature distillation loss
            weights: attention weights для каждого учителя
        """
        batch_size = labels.size(0)
        num_teachers = len(teacher_logits_list)
        device = labels.device

        # Вычисляем ошибки учителей (CE loss)
        criterion_ce = nn.CrossEntropyLoss(reduction='none')
        teacher_losses = []

        for logit_t in teacher_logits_list:
            loss_t = criterion_ce(logit_t, labels)  # (batch_size,)
            teacher_losses.append(loss_t.detach())

        # Stack losses: (num_teachers, batch_size)
        teacher_losses = torch.stack(teacher_losses, dim=0)

        # Compute attention weights (адаптивные веса)
        # w_i = (1 - softmax(loss)) / (M-1)
        # Используем mean loss для каждого учителя
        avg_losses = teacher_losses.mean(dim=1)  # (num_teachers,)

        # Нормализуем через softmax
        softmax_losses = F.softmax(avg_losses, dim=0)  # (num_teachers,)

        # Inverse: (1 - softmax_loss) - учитель с меньшей ошибкой получит больший вес
        weights = (1.0 - softmax_losses) / (num_teachers - 1) if num_teachers > 1 else torch.ones(1, device=device)
        weights = weights.clamp(min=1e-8)  # Избегаем нулевых весов

        # Feature distillation loss (MSE)
        mse_loss = 0.0

        for i, (student_feat, teacher_feat) in enumerate(zip(student_feat_list, teacher_feat_list)):
            # Вычисляем MSE между student и teacher features
            feat_loss = F.mse_loss(student_feat, teacher_feat, reduction='mean')

            # Взвешиваем по attention weight
            mse_loss += weights[i] * feat_loss

        # Нормализуем
        mse_loss = mse_loss / num_teachers

        return mse_loss, weights


class FidelityKL(nn.Module):
    """
    KL divergence для вычисления fidelity между моделями
    (Используется для отслеживания метрик, не для обучения)
    """
    def __init__(self, T=1.0):
        super(FidelityKL, self).__init__()
        self.T = T

    def forward(self, logits_s, logits_t):
        """Вычисляет KL(T || S)"""
        p_s = F.softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)

        # KL(T || S) = sum(p_t * log(p_t / p_s))
        kl = (p_t * (torch.log(p_t + 1e-10) - torch.log(p_s + 1e-10))).sum(dim=1).mean()

        return kl


def cosine_similarity_loss(feat_s, feat_t):
    """
    Cosine similarity loss для feature distillation
    Альтернатива MSE
    """
    # Нормализуем features
    feat_s_norm = F.normalize(feat_s, p=2, dim=1)
    feat_t_norm = F.normalize(feat_t, p=2, dim=1)

    # Cosine similarity
    cosine_sim = F.cosine_similarity(feat_s_norm, feat_t_norm, dim=1)

    # Loss: 1 - cosine_sim (максимизируем сходство)
    loss = (1 - cosine_sim).mean()

    return loss

