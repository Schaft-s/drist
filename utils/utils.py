#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для Knowledge Distillation
- Метрики (fidelity, top-1 agreement)
- Логирование
- Визуализация
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict


def compute_fidelity_metrics(student, teacher_models, data_loader, device, temperature=4.0):
    """
    Вычисляет все типы fidelity метрик

    Returns:
        dict с 4 типами метрик
    """
    student.eval()
    for teacher in teacher_models:
        teacher.eval()

    student_outputs_list = []
    teacher_outputs_list = [[] for _ in teacher_models]

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)

            student_out = student(data)
            student_outputs_list.append(student_out.cpu())

            for i, teacher in enumerate(teacher_models):
                teacher_out = teacher(data)
                teacher_outputs_list[i].append(teacher_out.cpu())

    # Concatenate
    student_outputs = torch.cat(student_outputs_list, dim=0)
    teacher_outputs_list = [torch.cat(outputs, dim=0) for outputs in teacher_outputs_list]

    # Softmax
    student_probs = F.softmax(student_outputs / temperature, dim=1)
    teacher_probs_list = [F.softmax(t / temperature, dim=1) for t in teacher_outputs_list]

    # 1. Individual Fidelity: KL(T_i || S)
    individual_fidelity = []
    for teacher_probs in teacher_probs_list:
        kl_div = (teacher_probs * (torch.log(teacher_probs + 1e-10) - 
                                    torch.log(student_probs + 1e-10))).sum(dim=1).mean().item()
        individual_fidelity.append(kl_div)

    # 2. Centroid Fidelity: KL(Centroid || S)
    centroid_probs = torch.stack(teacher_probs_list, dim=0).mean(dim=0)
    centroid_fidelity = (centroid_probs * (torch.log(centroid_probs + 1e-10) - 
                                           torch.log(student_probs + 1e-10))).sum(dim=1).mean().item()

    # 3. Teacher Diversity: KL(T_i || Centroid)
    teacher_diversity = []
    for teacher_probs in teacher_probs_list:
        kl_div = (teacher_probs * (torch.log(teacher_probs + 1e-10) - 
                                    torch.log(centroid_probs + 1e-10))).sum(dim=1).mean().item()
        teacher_diversity.append(kl_div)

    # 4. Pairwise Teacher Diversity: KL(T_i || T_j)
    teacher_pairwise_div = {}
    num_teachers = len(teacher_probs_list)
    for i in range(num_teachers):
        for j in range(i+1, num_teachers):
            key = f"T{i}_T{j}"
            kl_div = (teacher_probs_list[i] * (torch.log(teacher_probs_list[i] + 1e-10) - 
                                               torch.log(teacher_probs_list[j] + 1e-10))).sum(dim=1).mean().item()
            teacher_pairwise_div[key] = kl_div

    # 5. Top-1 Agreement
    student_preds = torch.argmax(student_outputs, dim=1)
    top1_agreement = {}

    for i, teacher_out in enumerate(teacher_outputs_list):
        teacher_preds = torch.argmax(teacher_out, dim=1)
        agreement = (student_preds == teacher_preds).float().mean().item() * 100
        top1_agreement[f"teacher_{i}"] = agreement

    return {
        'individual_fidelity': individual_fidelity,
        'centroid_fidelity': centroid_fidelity,
        'teacher_diversity': teacher_diversity,
        'teacher_pairwise_div': teacher_pairwise_div,
        'top1_agreement': top1_agreement
    }


class MetricsLogger:
    """Логирует метрики обучения"""

    def __init__(self):
        self.train_acc = []
        self.test_acc = []
        self.train_loss = []
        self.test_loss = []
        self.teacher_weights = defaultdict(list)
        self.individual_fidelity = defaultdict(list)
        self.centroid_fidelity = []
        self.teacher_diversity = defaultdict(list)
        self.teacher_pairwise_div = defaultdict(list)
        self.top1_agreement = defaultdict(list)

    def log(self, epoch=None, train_acc=None, test_acc=None, 
            train_loss=None, test_loss=None, teacher_weights=None,
            individual_fidelity=None, centroid_fidelity=None,
            teacher_diversity=None, teacher_pairwise_div=None, top1_agreement=None):

        if train_acc is not None:
            self.train_acc.append(train_acc)
        if test_acc is not None:
            self.test_acc.append(test_acc)
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if test_loss is not None:
            self.test_loss.append(test_loss)

        if teacher_weights is not None:
            for i, w in enumerate(teacher_weights):
                self.teacher_weights[f"teacher_{i}"].append(w)

        if individual_fidelity is not None:
            for i, fid in enumerate(individual_fidelity):
                self.individual_fidelity[f"teacher_{i}"].append(fid)

        if centroid_fidelity is not None:
            self.centroid_fidelity.append(centroid_fidelity)

        if teacher_diversity is not None:
            for i, div in enumerate(teacher_diversity):
                self.teacher_diversity[f"teacher_{i}"].append(div)

        if teacher_pairwise_div is not None:
            for key, val in teacher_pairwise_div.items():
                self.teacher_pairwise_div[key].append(val)

        if top1_agreement is not None:
            for name, agr in top1_agreement.items():
                self.top1_agreement[name].append(agr)

    def save(self, filepath):
        """Сохраняет метрики в JSON"""
        metrics = {
            'train_acc': self.train_acc,
            'test_acc': self.test_acc,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'teacher_weights': dict(self.teacher_weights),
            'individual_fidelity': dict(self.individual_fidelity),
            'centroid_fidelity': self.centroid_fidelity,
            'teacher_diversity': dict(self.teacher_diversity),
            'teacher_pairwise_div': dict(self.teacher_pairwise_div),
            'top1_agreement': dict(self.top1_agreement)
        }

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)


def evaluate_model(model, data_loader, criterion, device):
    """Оценивает модель на тестовом наборе"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    test_loss = test_loss / len(data_loader)

    return test_acc, test_loss


def plot_training_metrics(logger, teacher_names, save_path=None):
    """Строит базовые 9 графиков"""

    fig = plt.figure(figsize=(20, 12))
    epochs = list(range(1, len(logger.test_acc) + 1))

    # 1. Train/Test Accuracy
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(epochs, logger.train_acc, 'b-o', label='Train', linewidth=2, markersize=5)
    ax1.plot(epochs, logger.test_acc, 'r-s', label='Test', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Train/Test Accuracy', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Train/Test Loss
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(epochs, logger.train_loss, 'b-o', label='Train', linewidth=2, markersize=5)
    ax2.plot(epochs, logger.test_loss, 'r-s', label='Test', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Train/Test Loss', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Overfitting Gap
    ax3 = plt.subplot(3, 3, 3)
    gap = [t - tr for tr, t in zip(logger.train_acc, logger.test_acc)]
    ax3.plot(epochs, gap, 'g-^', linewidth=2, markersize=6)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Gap (%)', fontsize=11)
    ax3.set_title('Train-Test Overfitting Gap', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4-6. Individual Fidelity
    individual_fidelity = logger.individual_fidelity
    if individual_fidelity:
        fid_epochs = list(range(1, len(list(individual_fidelity.values())[0]) + 1))

        for idx in range(min(3, len(individual_fidelity))):
            ax = plt.subplot(3, 3, 4 + idx)
            key = f"teacher_{idx}"
            if key in individual_fidelity:
                ax.plot(fid_epochs, individual_fidelity[key], 'o-', linewidth=2, markersize=5)
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('KL Divergence', fontsize=11)
                ax.set_title(f'Individual Fidelity: {teacher_names[idx]}', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

    # 7. Centroid Fidelity
    ax7 = plt.subplot(3, 3, 7)
    if logger.centroid_fidelity:
        ax7.plot(fid_epochs, logger.centroid_fidelity, 'purple', 
                marker='D', linewidth=2, markersize=6)
        ax7.set_xlabel('Epoch', fontsize=11)
        ax7.set_ylabel('KL Divergence', fontsize=11)
        ax7.set_title('Centroid Fidelity', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

    # 8. Teacher Weights
    ax8 = plt.subplot(3, 3, 8)
    teacher_weights = logger.teacher_weights
    if teacher_weights:
        weight_epochs = list(range(1, len(list(teacher_weights.values())[0]) + 1))
        for i, (key, weights) in enumerate(teacher_weights.items()):
            ax8.plot(weight_epochs, weights, marker='o', label=teacher_names[i], linewidth=2)
        ax8.set_xlabel('Epoch', fontsize=11)
        ax8.set_ylabel('Weight', fontsize=11)
        ax8.set_title('Teacher Weights (CAMKD)', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)

    # 9. Top-1 Agreement
    ax9 = plt.subplot(3, 3, 9)
    top1_agreement = logger.top1_agreement
    if top1_agreement:
        agr_epochs = list(range(1, len(list(top1_agreement.values())[0]) + 1))
        for i, (key, agreements) in enumerate(top1_agreement.items()):
            ax9.plot(agr_epochs, agreements, marker='o', label=teacher_names[i], linewidth=2)
        ax9.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax9.set_xlabel('Epoch', fontsize=11)
        ax9.set_ylabel('Top-1 Agreement (%)', fontsize=11)
        ax9.set_title('Top-1 Agreement', fontsize=12, fontweight='bold')
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3)
        ax9.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Графики сохранены: {save_path}")

    plt.close()

