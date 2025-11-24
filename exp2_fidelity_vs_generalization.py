#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЭКСПЕРИМЕНТ 2: Зависимость Fidelity vs Generalization

Цель: Посмотреть на зависимость fidelity vs generalization через разные 
коэффициенты lambda перед KL loss.

Метрики:
- R(S_lambda) = CE(y, S_lambda)  - обобщаемость студента
- F_cent(S_lambda) = E[KL(hat{T} | S_lambda)] - centroid fidelity
- F_avg(S_lambda) = E[KL(avg_T | S_lambda)] - average fidelity

Процедура:
1. Загружаем 5 предобученных учителей из эксперимента 1
2. Для каждого lambda в [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
   - Обучаем новый студент (Centroid метод)
   - Вычисляем R(S_lambda), F_cent(S_lambda), F_avg(S_lambda)
3. Строим графики:
   - |R(S_lambda) - R(hat{T})| vs sqrt(F_cent(S_lambda))
   - F_avg(S_lambda) vs F_cent(S_lambda)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import sys
import json
import numpy as np
from collections import defaultdict

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from train_student.student_models import student_model_dict, count_parameters
from train_teachers.teacher_models import teacher_model_dict
from utils.distillation_losses import DistillKL
from utils.utils import compute_fidelity_metrics, evaluate_model


def load_config(config_path):
    """Загружает конфигурацию"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataloaders(dataset, batch_size):
    """Создает dataloaders"""
    if dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if dataset == 'MNIST':
            train_dataset = datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
        else:  # FashionMNIST
            train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def load_teachers(teacher_names, teacher_dir, dataset, device):
    """Загружает предобученных учителей"""
    teachers = []
    num_classes = 10
    
    for name in teacher_names:
        model = teacher_model_dict[name](num_classes=num_classes)
        checkpoint_path = f'{teacher_dir}/{dataset.lower()}/{name}.pth'
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            acc = checkpoint.get('test_acc', 0)
            print(f"✓ Загружен {name}: Test Acc = {acc:.2f}%")
        except Exception as e:
            print(f"✗ Ошибка загрузки {name}: {e}")
            return None
        
        teachers.append((model, name))
    
    return teachers


def compute_centroid_fidelity(student, teachers, test_loader, device, temperature):
    """
    Вычисляет Centroid Fidelity: KL(avg_logits_teachers | student_logits)
    """
    student.eval()
    
    total_kl = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Логиты студента
            logits_s = student(data)
            
            # Логиты учителей
            logits_t_list = []
            for teacher, _ in teachers:
                if hasattr(teacher, 'is_feat'):
                    _, logits_t = teacher(data, is_feat=True)
                else:
                    logits_t = teacher(data)
                logits_t_list.append(logits_t)
            
            # Среднее распределение учителей (центроид)
            avg_logits_t = sum(logits_t_list) / len(logits_t_list)
            
            # KL divergence
            probs_t = F.softmax(avg_logits_t / temperature, dim=1)
            log_probs_s = F.log_softmax(logits_s / temperature, dim=1)
            kl = F.kl_div(log_probs_s, probs_t, reduction='batchmean')
            
            total_kl += kl.item()
            num_batches += 1
    
    return total_kl / num_batches


def compute_average_fidelity(student, teachers, test_loader, device, temperature):
    """
    Вычисляет Average Fidelity: среднее KL со всеми учителями
    """
    student.eval()
    
    total_kl = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Логиты студента
            logits_s = student(data)
            
            batch_kl = 0.0
            for teacher, _ in teachers:
                if hasattr(teacher, 'is_feat'):
                    _, logits_t = teacher(data, is_feat=True)
                else:
                    logits_t = teacher(data)
                
                # KL divergence
                probs_t = F.softmax(logits_t / temperature, dim=1)
                log_probs_s = F.log_softmax(logits_s / temperature, dim=1)
                kl = F.kl_div(log_probs_s, probs_t, reduction='batchmean')
                batch_kl += kl.item()
            
            total_kl += batch_kl / len(teachers)
            num_batches += 1
    
    return total_kl / num_batches


def train_student_with_lambda(lambda_kd, student, teachers, train_loader, test_loader,
                              device, config, epochs=20):
    """
    Обучает студента с фиксированным lambda перед KL loss
    
    Loss = CE(y, S) + lambda * KL(avg_T | S)
    """
    
    student = student.to(device)
    student.train()
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillKL(config['temperature'])
    optimizer = optim.Adam(student.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print(f"\n  Обучение с lambda={lambda_kd}...")
    
    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Логиты студента
            logits_s = student(data)
            loss_cls = criterion_cls(logits_s, labels)
            
            # Средний логит учителей
            with torch.no_grad():
                logits_t_list = []
                for teacher, _ in teachers:
                    if hasattr(teacher, 'is_feat'):
                        _, logits_t = teacher(data, is_feat=True)
                    else:
                        logits_t = teacher(data)
                    logits_t_list.append(logits_t)
                
                avg_logits_t = sum(logits_t_list) / len(logits_t_list)
            
            # KL loss
            loss_kd = criterion_kd(logits_s, avg_logits_t)
            
            # Общий loss
            loss = loss_cls + lambda_kd * loss_kd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits_s.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                acc = 100. * train_correct / train_total
                print(f"    E{epoch+1} B{batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.4f}, Acc: {acc:.1f}%", end='\r')
        
        scheduler.step()
    
    print(f"\n  ✓ Обучение завершено с lambda={lambda_kd}")
    
    return student


def experiment_fidelity_vs_generalization(config):
    """
    Главная функция: Эксперимент 2
    Fidelity vs Generalization с разными lambda
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    print(f"Датасет: {config['dataset']}")
    
    # Data loaders
    train_loader, test_loader = get_dataloaders(config['dataset'], config['batch_size'])
    print(f"Обучающая выборка: {len(train_loader.dataset)}")
    print(f"Тестовая выборка: {len(test_loader.dataset)}")
    
    # Загружаем учителей из эксперимента 1
    teacher_names_config = config['teacher_names']
    teachers = load_teachers(teacher_names_config, config['teacher_dir'], 
                            config['dataset'], device)
    
    if teachers is None:
        print("\n✗ Не удалось загрузить учителей!")
        return
    
    print(f"✓ Загружено учителей: {len(teachers)}")
    
    # Вычисляем baseline: точность учителей (центроид)
    print("\nВычисляем baseline метрики для центроида учителей...")
    
    with torch.no_grad():
        teacher_accs = []
        for teacher, name in teachers:
            test_acc, _ = evaluate_model(teacher, test_loader, nn.CrossEntropyLoss(), device)
            teacher_accs.append(test_acc)
            print(f"  {name}: {test_acc:.2f}%")
        
        avg_teacher_acc = np.mean(teacher_accs)
        print(f"  Средняя точность учителей: {avg_teacher_acc:.2f}%")
    
    # Create save directory
    save_dir = f'{config["save_dir"]}/{config["dataset"].lower()}/exp2_fidelity_vs_gen2'
    os.makedirs(save_dir, exist_ok=True)
    
    # ========================================
    # Эксперимент: разные lambda
    # ========================================
    
    lambdas = [0.0, 0.1, 0.01, 0.25, 0.025, 100.0, 1000]
    results = defaultdict(dict)
    
    print("\n" + "="*70)
    print("ЭКСПЕРИМЕНТ: Fidelity vs Generalization")
    print("="*70)
    
    for lambda_kd in lambdas:
        print(f"\n{'='*70}")
        print(f"Lambda = {lambda_kd}")
        print(f"{'='*70}")
        
        # Создаём и обучаем студента
        student_model_name = config.get('student_model', 'student') # Читаем из конфига
        student = student_model_dict[student_model_name](num_classes=10).to(device)
        print(f"Студент: {count_parameters(student):,} параметров")
        
        # Обучаем
        student = train_student_with_lambda(
            lambda_kd, student, teachers, train_loader, test_loader,
            device, config, epochs=config.get('exp2_epochs', 20)
        )
        
        # Вычисляем метрики на тестовой выборке
        print(f"\n  Вычисляем метрики...")
        
        # 1. Generalization: R(S_lambda) = CE(y, S_lambda)
        test_acc, test_loss = evaluate_model(student, test_loader, nn.CrossEntropyLoss(), device)
        R_S = test_loss  # Cross-entropy loss
        generalization_gap = np.abs(test_acc - avg_teacher_acc)
        
        print(f"    Test Acc: {test_acc:.2f}%")
        print(f"    Test Loss (R): {R_S:.4f}")
        print(f"    Generalization gap: |{test_acc:.2f}% - {avg_teacher_acc:.2f}%| = {generalization_gap:.2f}%")
        
        # 2. Centroid Fidelity: F_cent(S_lambda)
        F_cent = compute_centroid_fidelity(student, teachers, test_loader, device, 
                                          config['temperature'])
        print(f"    Centroid Fidelity: {F_cent:.4f}")
        
        # 3. Average Fidelity: F_avg(S_lambda)
        F_avg = compute_average_fidelity(student, teachers, test_loader, device,
                                         config['temperature'])
        print(f"    Average Fidelity: {F_avg:.4f}")
        
        # Сохраняем результаты
        results[lambda_kd] = {
            'test_acc': test_acc,
            'test_loss': R_S,
            'generalization_gap': generalization_gap,
            'centroid_fidelity': F_cent,
            'average_fidelity': F_avg,
            'sqrt_F_cent': np.sqrt(F_cent)
        }
        
        # Сохраняем модель студента
        save_path = f'{save_dir}/student_lambda_{lambda_kd}.pth'
        torch.save({
            'model_state_dict': student.state_dict(),
            'test_acc': test_acc,
            'lambda': lambda_kd
        }, save_path)
        print(f"    ✓ Модель сохранена: {save_path}")
    
    # ========================================
    # Результаты и анализ
    # ========================================
    
    print("\n" + "="*70)
    print("ИТОГИ ЭКСПЕРИМЕНТА")
    print("="*70)
    
    print("\nЛямбда | Test Acc | R(S_λ) | Generalization Gap | F_cent | F_avg | sqrt(F_cent)")
    print("-" * 85)
    
    for lambda_kd in lambdas:
        res = results[lambda_kd]
        print(f"{lambda_kd:6.1f} | {res['test_acc']:7.2f}% | {res['test_loss']:6.4f} | "
              f"{res['generalization_gap']:8.2f}% | {res['centroid_fidelity']:6.4f} | "
              f"{res['average_fidelity']:6.4f} | {res['sqrt_F_cent']:6.4f}")
    
    # ========================================
    # Построение графиков
    # ========================================
    
    print(f"\nПостроение графиков...")
    
    try:
        import matplotlib.pyplot as plt
        
        lambdas_list = sorted(lambdas)
        
        # График 1: |R(S_lambda) - R(hat{T})| vs sqrt(F_cent)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        gen_gaps = [results[l]['generalization_gap'] for l in lambdas_list]
        sqrt_f_cents = [results[l]['sqrt_F_cent'] for l in lambdas_list]
        f_avgs = [results[l]['average_fidelity'] for l in lambdas_list]
        f_cents = [results[l]['centroid_fidelity'] for l in lambdas_list]
        
        # График 1
        ax1 = axes[0]
        scatter1 = ax1.scatter(sqrt_f_cents, gen_gaps, s=200, c=lambdas_list, 
                              cmap='viridis', edgecolors='black', linewidth=2, alpha=0.7)
        
        for i, l in enumerate(lambdas_list):
            ax1.annotate(f'λ={l}', (sqrt_f_cents[i], gen_gaps[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax1.set_xlabel('√F_cent(S_λ)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('|R(S_λ) - R(T̂)|, %', fontsize=12, fontweight='bold')
        ax1.set_title('Generalization Gap vs Centroid Fidelity', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='λ')
        
        # График 2: F_avg vs F_cent
        ax2 = axes[1]
        scatter2 = ax2.scatter(f_cents, f_avgs, s=200, c=lambdas_list,
                              cmap='viridis', edgecolors='black', linewidth=2, alpha=0.7)
        
        for i, l in enumerate(lambdas_list):
            ax2.annotate(f'λ={l}', (f_cents[i], f_avgs[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax2.set_xlabel('F_cent(S_λ)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F_avg(S_λ)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Fidelity vs Centroid Fidelity', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='λ')
        
        plt.tight_layout()
        plot_path = f'{save_dir}/fidelity_vs_generalization.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Графики сохранены: {plot_path}")
        plt.close()
        
        # Дополнительные графики
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Test Accuracy vs Lambda
        test_accs = [results[l]['test_acc'] for l in lambdas_list]
        ax = axes[0, 0]
        ax.plot(lambdas_list, test_accs, 'o-', linewidth=2, markersize=8, label='Student', color='blue')
        ax.axhline(y=avg_teacher_acc, color='red', linestyle='--', linewidth=2, label='Avg Teacher')
        ax.set_xlabel('λ', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test Accuracy, %', fontsize=11, fontweight='bold')
        ax.set_title('Test Accuracy vs λ', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Centroid Fidelity vs Lambda
        ax = axes[0, 1]
        ax.plot(lambdas_list, f_cents, 's-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('λ', fontsize=11, fontweight='bold')
        ax.set_ylabel('F_cent', fontsize=11, fontweight='bold')
        ax.set_title('Centroid Fidelity vs λ', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Average Fidelity vs Lambda
        ax = axes[1, 0]
        ax.plot(lambdas_list, f_avgs, '^-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('λ', fontsize=11, fontweight='bold')
        ax.set_ylabel('F_avg', fontsize=11, fontweight='bold')
        ax.set_title('Average Fidelity vs λ', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Generalization Gap vs Lambda
        ax = axes[1, 1]
        ax.plot(lambdas_list, gen_gaps, 'd-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('λ', fontsize=11, fontweight='bold')
        ax.set_ylabel('Generalization Gap, %', fontsize=11, fontweight='bold')
        ax.set_title('Generalization Gap vs λ', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot2_path = f'{save_dir}/metrics_vs_lambda.png'
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        print(f"✓ Графики сохранены: {plot2_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Ошибка при построении графиков: {e}")
    
    # Сохраняем результаты в JSON
    results_json = {str(k): v for k, v in results.items()}
    results_path = f'{save_dir}/experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Результаты сохранены: {results_path}")
    
    print(f"\n✓ Эксперимент завершен!")
    print(f"📁 Результаты сохранены в: {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Эксперимент 2: Fidelity vs Generalization')
    parser.add_argument('--config', type=str, required=True,
                       help='Путь к конфигурационному файлу YAML')
    args = parser.parse_args()
    
    config = load_config(args.config)
    experiment_fidelity_vs_generalization(config)
