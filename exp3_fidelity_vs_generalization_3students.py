#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, argparse, yaml, sys, json
import numpy as np
from collections import defaultdict

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from train_student.student_models import student_model_dict, count_parameters
from train_teachers.teacher_models import teacher_model_dict
from utils.distillation_losses import DistillKL
from utils.utils import evaluate_model


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_dataloaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if dataset == 'MNIST':
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        else:
            train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def load_teachers(teacher_names, teacher_dir, dataset, device):
    teachers = []
    for name in teacher_names:
        model = teacher_model_dict[name](num_classes=10)
        ckpt_path = f'{teacher_dir}/{dataset.lower()}/{name}.pth'
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        acc = checkpoint.get('test_acc', 0)
        print(f"✓ Загружен {name}: Test Acc = {acc:.2f}%")
        teachers.append((model, name))
    return teachers


def compute_teacher_centroid_loss(teachers, test_loader, device):
    """R(Ť) = CE(y, Ť) где Ť — центроид логитов учителей."""
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            logits_t_list = []
            for teacher, _ in teachers:
                if hasattr(teacher, 'is_feat'):
                    _, lt = teacher(data, is_feat=True)
                else:
                    lt = teacher(data)
                logits_t_list.append(lt)
            avg_logits_t = sum(logits_t_list) / len(logits_t_list)
            loss_batch = ce(avg_logits_t, labels)
            total_loss += loss_batch.item()
            total_samples += labels.size(0)
    return total_loss / total_samples


def compute_centroid_fidelity(student, teachers, test_loader, device, temperature):
    """F_cent(S) = E KL(Ť | S), усреднённый по батчам."""
    student.eval()
    total_kl, num_batches = 0.0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            logits_s = student(data)

            logits_t_list = []
            for teacher, _ in teachers:
                if hasattr(teacher, 'is_feat'):
                    _, lt = teacher(data, is_feat=True)
                else:
                    lt = teacher(data)
                logits_t_list.append(lt)
            avg_logits_t = sum(logits_t_list) / len(logits_t_list)

            probs_t = F.softmax(avg_logits_t / temperature, dim=1)
            log_probs_s = F.log_softmax(logits_s / temperature, dim=1)
            kl = F.kl_div(log_probs_s, probs_t, reduction='batchmean')
            total_kl += kl.item()
            num_batches += 1
    return total_kl / num_batches


def compute_average_fidelity(student, teachers, test_loader, device, temperature):
    """F_avg(S) = E_i E KL(T_i | S)."""
    student.eval()
    total_kl, num_batches = 0.0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            logits_s = student(data)
            batch_kl = 0.0
            for teacher, _ in teachers:
                if hasattr(teacher, 'is_feat'):
                    _, lt = teacher(data, is_feat=True)
                else:
                    lt = teacher(data)
                probs_t = F.softmax(lt / temperature, dim=1)
                log_probs_s = F.log_softmax(logits_s / temperature, dim=1)
                kl = F.kl_div(log_probs_s, probs_t, reduction='batchmean')
                batch_kl += kl.item()
            total_kl += batch_kl / len(teachers)
            num_batches += 1
    return total_kl / num_batches


def train_student_with_lambda(lambda_kd, student, teachers, train_loader, device, config, epochs):
    student.to(device).train()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillKL(config['temperature'])
    optimizer = optim.Adam(student.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n  Обучение с lambda={lambda_kd}...")
    for epoch in range(epochs):
        student.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            logits_s = student(data)
            loss_cls = criterion_cls(logits_s, labels)

            with torch.no_grad():
                logits_t_list = []
                for teacher, _ in teachers:
                    if hasattr(teacher, 'is_feat'):
                        _, lt = teacher(data, is_feat=True)
                    else:
                        lt = teacher(data)
                    logits_t_list.append(lt)
                avg_logits_t = sum(logits_t_list) / len(logits_t_list)

            loss_kd = criterion_kd(logits_s, avg_logits_t)
            loss = loss_cls + lambda_kd * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = logits_s.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                acc = 100. * correct / total
                print(f"    E{epoch+1} B{batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.4f}, Acc: {acc:.1f}%", end='\r')
        scheduler.step()
    print(f"\n  ✓ Обучение завершено с lambda={lambda_kd}")
    return student


def experiment_fidelity_vs_generalization_3students(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    print(f"Датасет: {config['dataset']}")

    train_loader, test_loader = get_dataloaders(config['dataset'], config['batch_size'])
    print(f"Обучающая выборка: {len(train_loader.dataset)}")
    print(f"Тестовая выборка: {len(test_loader.dataset)}")

    teachers = load_teachers(config['teacher_names'], config['teacher_dir'],
                             config['dataset'], device)
    print(f"✓ Загружено учителей: {len(teachers)}")

    # Baseline: R(T̂)
    print("\nВычисляем R(Ť) для центроида учителей...")
    R_T = compute_teacher_centroid_loss(teachers, test_loader, device)
    print(f"  R(Ť) = {R_T:.4f}")

    save_dir = f'{config["save_dir"]}/{config["dataset"].lower()}/exp3_fidelity_vs_gen_3students'
    os.makedirs(save_dir, exist_ok=True)

    lambdas = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 100.0, 1000.0]
    student_models = ['micro_student', 'tiny_student', 'student']

    all_results = {}  # {student_model: {lambda: metrics}}

    for sm_name in student_models:
        if sm_name not in student_model_dict:
            print(f"✗ Модель {sm_name} не найдена, пропускаю")
            continue

        print("\n" + "="*70)
        print(f"СТУДЕНТ: {sm_name}")
        print("="*70)

        results = {}
        all_results[sm_name] = results

        for lambda_kd in lambdas:
            print(f"\n{'-'*70}\nLambda = {lambda_kd}\n{'-'*70}")
            student = student_model_dict[sm_name](num_classes=10)
            print(f"Студент: {count_parameters(student):,} параметров")

            student = train_student_with_lambda(
                lambda_kd, student, teachers, train_loader,
                device, config, epochs=config.get('exp2_epochs', 20)
            )

            print("\n  Вычисляем метрики...")
            test_acc, test_loss = evaluate_model(student, test_loader, nn.CrossEntropyLoss(), device)
            R_S = test_loss
            loss_gap = abs(R_S - R_T)

            F_cent = compute_centroid_fidelity(student, teachers, test_loader, device, config['temperature'])
            F_avg = compute_average_fidelity(student, teachers, test_loader, device, config['temperature'])

            results[lambda_kd] = dict(
                test_acc=test_acc,
                R_S=R_S,
                loss_gap=loss_gap,
                F_cent=F_cent,
                F_avg=F_avg,
                sqrt_F_cent=np.sqrt(F_cent),
            )

            save_path = f'{save_dir}/student_{sm_name}_lambda_{lambda_kd}.pth'
            torch.save({'model_state_dict': student.state_dict(),
                        'test_acc': test_acc,
                        'lambda': lambda_kd},
                       save_path)
            print(f"    ✓ Модель сохранена: {save_path}")

        # Печать таблички по студенту
        print("\nИТОГИ ДЛЯ", sm_name)
        print("Lambda | TestAcc | R(S)   | |R(S)-R(T)| | F_cent | F_avg | sqrt(F_cent)")
        print("-"*80)
        for lmb in lambdas:
            r = results[lmb]
            print(f"{lmb:6.3f} | {r['test_acc']:7.2f}% | {r['R_S']:6.4f} | "
                  f"{r['loss_gap']:10.4f} | {r['F_cent']:6.4f} | {r['F_avg']:6.4f} | {r['sqrt_F_cent']:6.4f}")

        # Графики для каждого студента отдельно
        try:
            import matplotlib.pyplot as plt
            lmb_list = lambdas
            gaps = [results[l]['loss_gap'] for l in lmb_list]
            sqrt_fc = [results[l]['sqrt_F_cent'] for l in lmb_list]
            favg = [results[l]['F_avg'] for l in lmb_list]
            fcent = [results[l]['F_cent'] for l in lmb_list]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax1 = axes[0]
            sc1 = ax1.scatter(sqrt_fc, gaps, c=lmb_list, s=200, cmap='viridis',
                              edgecolors='black', linewidth=2, alpha=0.8)
            for i, l in enumerate(lmb_list):
                ax1.annotate(f'λ={l}', (sqrt_fc[i], gaps[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
            ax1.set_xlabel('√F_cent(S_λ)')
            ax1.set_ylabel('|R(S_λ) - R(Ť)|')
            ax1.set_title(f'Loss Gap vs Fidelity ({sm_name})')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(sc1, ax=ax1, label='λ')

            ax2 = axes[1]
            sc2 = ax2.scatter(fcent, favg, c=lmb_list, s=200, cmap='viridis',
                              edgecolors='black', linewidth=2, alpha=0.8)
            for i, l in enumerate(lmb_list):
                ax2.annotate(f'λ={l}', (fcent[i], favg[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
            ax2.set_xlabel('F_cent(S_λ)')
            ax2.set_ylabel('F_avg(S_λ)')
            ax2.set_title(f'F_avg vs F_cent ({sm_name})')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(sc2, ax=ax2, label='λ')

            plt.tight_layout()
            plot_path = f'{save_dir}/fidelity_vs_generalization_{sm_name}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Графики для {sm_name} сохранены: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"⚠️ Ошибка при построении графиков для {sm_name}: {e}")

    # Сохраняем всё в JSON
    out_path = f'{save_dir}/experiment3_results_3students.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Все результаты сохранены: {out_path}")
    print(f"📁 Папка результатов: {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Эксперимент 3: Fidelity vs Generalization, 3 студента')
    parser.add_argument('--config', type=str, required=True,
                        help='Путь к конфигурационному файлу YAML (FashionMNIST)')
    args = parser.parse_args()
    cfg = load_config(args.config)
    experiment_fidelity_vs_generalization_3students(cfg)
