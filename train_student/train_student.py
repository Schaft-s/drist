#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение студента двумя методами:
1. CAMKD - Cross-teacher Attentive Multi-teacher KD
2. Vanilla - Обычная дистилляция от усреднённого учителя

Поддерживает параллельное обучение для честного сравнения
Использует конфигурационные файлы YAML
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

# Добавляем родительскую директорию в путь для импорта
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from train_student.student_models import student_model_dict
from train_student.convreg import ConvReg
from train_teachers.teacher_models import teacher_model_dict, count_parameters
from utils.distillation_losses import DistillKL, CAMKD
from utils.utils import (compute_fidelity_metrics, MetricsLogger, 
                         plot_training_metrics, evaluate_model)


def load_config(config_path):
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser('Обучение студента с сравнением методов')
    parser.add_argument('--config', type=str, required=True,
                       help='Путь к конфигурационному файлу YAML')
    return parser.parse_args()


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
    fidelity_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2)

    return train_loader, test_loader, fidelity_loader


def load_teachers(teacher_names, teacher_dir, dataset, device, use_pretrained=False):
    """Загружает учителей"""
    teachers = []
    num_classes = 10

    for name in teacher_names:
        model = teacher_model_dict[name](num_classes=num_classes)
        checkpoint_path = f'{teacher_dir}/{dataset.lower()}/{name}.pth'

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            acc = checkpoint.get('test_acc', 0)
            print(f"✓ Загружен {name}: Test Acc = {acc:.2f}%")
        except FileNotFoundError:
            print(f"✗ Файл {checkpoint_path} не найден!")
            return None

        teachers.append((model, name, acc))

    return teachers


def train_epoch_camkd(epoch, student, teachers, regress_list, train_loader,
                     criterion_cls, criterion_kd, criterion_camkd, optimizer, device, config):
    """Обучение эпохи с CAMKD"""
    student.train()
    for regress in regress_list:
        regress.train()

    train_loss = 0.0
    correct = 0
    total = 0
    teacher_weights_accum = [0.0] * len(teachers)
    num_batches = 0
    top1_agreement_accum = {name: 0.0 for _, name, _ in teachers}

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward student
        feat_s, logit_s = student(data, is_feat=True)

        # Forward teachers
        feat_t_list = []
        logit_t_list = []
        with torch.no_grad():
            for teacher_model, _, _ in teachers:
                feat_t, logit_t = teacher_model(data, is_feat=True)
                feat_t_list.append(feat_t)
                logit_t_list.append(logit_t)

        # Transform student features
        trans_feat_s_list = []
        mid_feat_t_list = []
        for feat_t, regress in zip(feat_t_list, regress_list):
            trans_feat_s = regress(feat_s[config['hint_layer']])
            trans_feat_s_list.append(trans_feat_s)
            mid_feat_t_list.append(feat_t[config['hint_layer']])

        # Losses
        loss_cls = criterion_cls(logit_s, labels)
        avg_logit_t = sum(logit_t_list) / len(logit_t_list)
        loss_kd = criterion_kd(logit_s, avg_logit_t)
        loss_feat, teacher_weights = criterion_camkd(
            trans_feat_s_list, mid_feat_t_list, logit_t_list, labels
        )

        loss = loss_cls + config['alpha'] * loss_kd + config['beta'] * loss_feat

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logit_s.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Top-1 Agreement
        for i, (_, name, _) in enumerate(teachers):
            teacher_preds = torch.argmax(logit_t_list[i], dim=1)
            agreement = (predicted == teacher_preds).float().sum().item()
            top1_agreement_accum[name] += agreement

        for i, w in enumerate(teacher_weights):
            teacher_weights_accum[i] += w.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f'  E{epoch+1} B{batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')

    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    avg_teacher_weights = [w / num_batches for w in teacher_weights_accum]
    avg_top1_agreement = {name: (top1_agreement_accum[name] / total * 100) 
                         for name in top1_agreement_accum}

    return train_acc, avg_loss, avg_teacher_weights, avg_top1_agreement


def train_epoch_vanilla(epoch, student, teachers, train_loader,
                       criterion_cls, criterion_kd, optimizer, device, config):
    """Обучение эпохи с Vanilla KD"""
    student.train()

    train_loss = 0.0
    correct = 0
    total = 0
    top1_agreement_accum = {name: 0.0 for _, name, _ in teachers}

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward student
        logit_s = student(data)

        # Forward teachers
        with torch.no_grad():
            logit_t_list = []
            for teacher_model, _, _ in teachers:
                logit_t = teacher_model(data)
                logit_t_list.append(logit_t)
            avg_logit_t = sum(logit_t_list) / len(logit_t_list)

        # Losses
        loss_cls = criterion_cls(logit_s, labels)
        loss_kd = criterion_kd(logit_s, avg_logit_t)
        loss = loss_cls + config['alpha'] * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logit_s.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Top-1 Agreement
        for i, (_, name, _) in enumerate(teachers):
            teacher_preds = torch.argmax(logit_t_list[i], dim=1)
            agreement = (predicted == teacher_preds).float().sum().item()
            top1_agreement_accum[name] += agreement

        if batch_idx % 100 == 0:
            print(f'  E{epoch+1} B{batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')

    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    teacher_weights = [1.0/len(teachers)] * len(teachers)
    avg_top1_agreement = {name: (top1_agreement_accum[name] / total * 100) 
                         for name in top1_agreement_accum}

    return train_acc, avg_loss, teacher_weights, avg_top1_agreement


def train_student_method(method_name, student, teachers, teacher_names, 
                        train_loader, test_loader, fidelity_loader, device, config):
    """Обучение студента одним методом"""

    logger = MetricsLogger()
    best_acc = 0.0

    print(f"\n{'='*70}")
    print(f"Обучение студента: {method_name}")
    print(f"Параметров: {count_parameters(student):,}")
    print(f"{'='*70}\n")

    if method_name == 'CAMKD':
        # Setup CAMKD
        regress_list = nn.ModuleList()

        img_size = 32 if config['dataset'] == 'CIFAR10' else 28
        channels = 3 if config['dataset'] == 'CIFAR10' else 1
        dummy_input = torch.randn(2, channels, img_size, img_size)

        student.to(device)
        feat_s, _ = student(dummy_input, is_feat=True)

        for teacher_model, _, _ in teachers:
            feat_t, _ = teacher_model(dummy_input, is_feat=True)
            s_shape = [dummy_input.shape[0]] + list(feat_s[config['hint_layer']].shape[1:])
            t_shape = [dummy_input.shape[0]] + list(feat_t[config['hint_layer']].shape[1:])
            regress = ConvReg(s_shape, t_shape).to(device)
            regress_list.append(regress)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_kd = DistillKL(config['temperature'])
        criterion_camkd = CAMKD()

        trainable_params = list(student.parameters())
        for regress in regress_list:
            trainable_params += list(regress.parameters())

        optimizer = optim.Adam(trainable_params, lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(config['epochs']):
            train_acc, train_loss, teacher_weights, top1_agreement = train_epoch_camkd(
                epoch, student, teachers, regress_list, train_loader,
                criterion_cls, criterion_kd, criterion_camkd, optimizer, device, config
            )

            test_acc, test_loss = evaluate_model(student, test_loader, criterion_cls, device)

            print(f'Epoch {epoch+1}: Train={train_acc:.2f}% Test={test_acc:.2f}%')

            if (epoch + 1) % config['fidelity_freq'] == 0:
                print(f"  Вычисление fidelity метрик...")
                teacher_models = [t[0] for t in teachers]
                fidelity_metrics = compute_fidelity_metrics(
                    student, teacher_models, fidelity_loader, device, config['temperature']
                )

                logger.log(
                    epoch=epoch+1,
                    individual_fidelity=fidelity_metrics['individual_fidelity'],
                    centroid_fidelity=fidelity_metrics['centroid_fidelity'],
                    teacher_diversity=fidelity_metrics['teacher_diversity'],
                    teacher_pairwise_div=fidelity_metrics['teacher_pairwise_div'],
                    top1_agreement=fidelity_metrics['top1_agreement']
                )

            logger.log(
                epoch=epoch+1,
                train_acc=train_acc,
                train_loss=train_loss,
                test_acc=test_acc,
                test_loss=test_loss,
                teacher_weights=teacher_weights
            )

            if test_acc > best_acc:
                best_acc = test_acc
                save_path = f'{config["save_dir"]}/{config["dataset"].lower()}/student_{method_name.lower()}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'test_acc': test_acc,
                }, save_path)
                print(f'✓ Сохранена модель: {test_acc:.2f}%\n')

            scheduler.step()

    else:  # Vanilla
        criterion_cls = nn.CrossEntropyLoss()
        criterion_kd = DistillKL(config['temperature'])
        student = student.to(device)

        optimizer = optim.Adam(student.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(config['epochs']):
            train_acc, train_loss, teacher_weights, top1_agreement = train_epoch_vanilla(
                epoch, student, teachers, train_loader,
                criterion_cls, criterion_kd, optimizer, device, config
            )

            test_acc, test_loss = evaluate_model(student, test_loader, criterion_cls, device)

            print(f'Epoch {epoch+1}: Train={train_acc:.2f}% Test={test_acc:.2f}%')

            if (epoch + 1) % config['fidelity_freq'] == 0:
                print(f"  Вычисление fidelity метрик...")
                teacher_models = [t[0] for t in teachers]
                fidelity_metrics = compute_fidelity_metrics(
                    student, teacher_models, fidelity_loader, device, config['temperature']
                )

                logger.log(
                    epoch=epoch+1,
                    individual_fidelity=fidelity_metrics['individual_fidelity'],
                    centroid_fidelity=fidelity_metrics['centroid_fidelity'],
                    teacher_diversity=fidelity_metrics['teacher_diversity'],
                    teacher_pairwise_div=fidelity_metrics['teacher_pairwise_div'],
                    top1_agreement=fidelity_metrics['top1_agreement']
                )

            logger.log(
                epoch=epoch+1,
                train_acc=train_acc,
                train_loss=train_loss,
                test_acc=test_acc,
                test_loss=test_loss,
                teacher_weights=teacher_weights
            )

            if test_acc > best_acc:
                best_acc = test_acc
                save_path = f'{config["save_dir"]}/{config["dataset"].lower()}/student_{method_name.lower()}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'test_acc': test_acc,
                }, save_path)
                print(f'✓ Сохранена модель: {test_acc:.2f}%\n')

            scheduler.step()

    # Save metrics
    metrics_path = f'{config["save_dir"]}/{config["dataset"].lower()}/metrics_{method_name.lower()}.json'
    logger.save(metrics_path)
    print(f"✓ Метрики сохранены: {metrics_path}")

    # Plot
    plot_path = f'{config["save_dir"]}/{config["dataset"].lower()}/plots_{method_name.lower()}.png'
    plot_training_metrics(logger, teacher_names, save_path=plot_path)

    return best_acc, logger


def compare_methods(config):
    """Сравнение двух методов дистилляции"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    print(f"Датасет: {config['dataset']}")

    # Data loaders
    train_loader, test_loader, fidelity_loader = get_dataloaders(config['dataset'], config['batch_size'])

    # Teachers
    teacher_names = config['teacher_names']
    teachers = load_teachers(teacher_names, config['teacher_dir'], config['dataset'], 
                            device, config.get('use_pretrained', False))

    if teachers is None:
        print("\n✗ Не удалось загрузить учителей!")
        print("Сначала запустите: python train_teachers/train_teachers.py --config train_teachers/configs/...")
        return

    results = {}

    # Train with each method
    methods = []
    if config['methods'] in ['camkd', 'both']:
        methods.append('CAMKD')
    if config['methods'] in ['vanilla', 'both']:
        methods.append('Vanilla')

    for method in methods:
        # Create fresh student
        num_classes = 10
        student = student_model_dict['student'](num_classes=num_classes)

        # Train
        best_acc, logger = train_student_method(
            method, student, teachers, teacher_names,
            train_loader, test_loader, fidelity_loader, device, config
        )

        results[method] = {
            'best_acc': best_acc,
            'logger': logger
        }

    # Print comparison
    if len(methods) > 1:
        print(f"\n{'='*70}")
        print("СРАВНЕНИЕ МЕТОДОВ")
        print(f"{'='*70}")
        print(f"Учителя:")
        for _, name, acc in teachers:
            print(f"  {name}: {acc:.2f}%")
        print(f"\nСтуденты:")
        for method in methods:
            print(f"  {method}: {results[method]['best_acc']:.2f}%")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    args = parse_args()

    # Загружаем конфигурацию
    config = load_config(args.config)

    compare_methods(config)

