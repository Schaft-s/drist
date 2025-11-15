#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Параллельное обучение 4 студентов на одинаковых батчах
4 метода дистилляции в одном цикле:
1. Centroid    - KD от среднего учителя
2. Best        - KD от лучшего учителя на батче
3. Median      - KD от медианного учителя
4. CAMKD       - адаптивные веса + feature distillation (правильная реализация)

КЛЮЧЕВОЕ ОТЛИЧИЕ: все 4 студента обновляются на одних и тех же батчах
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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from train_student.student_models import student_model_dict, count_parameters
from train_student.convreg import ConvReg
from train_teachers.teacher_models import teacher_model_dict
from utils.distillation_losses import DistillKL, CAMKD
from utils.utils import compute_fidelity_metrics, MetricsLogger, evaluate_model


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
    fidelity_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2)
    
    return train_loader, test_loader, fidelity_loader


def load_teachers(teacher_names, teacher_dir, dataset, device):
    """Загружает обученных учителей"""
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
        except Exception as e:
            print(f"✗ Ошибка загрузки {name}: {e}")
            return None
        
        teachers.append((model, name))
    
    return teachers


def train_epoch_parallel_4methods(epoch, students, teachers, regress_lists, train_loader,
                                  criterion_cls, criterion_kd, criterion_camkd,
                                  optimizers, device, config):
    """
    Обучает 4 студентов ПАРАЛЛЕЛЬНО на одинаковых батчах
    
    students: {'centroid': model, 'best': model, 'median': model, 'camkd': model}
    teachers: [(model, name), ...]
    """
    
    # Переводим все модели в режим обучения
    for student in students.values():
        student.train()
    
    for regress_list in regress_lists.values():
        for regress in regress_list:
            regress.train()
    
    # Инициализируем метрики для каждого студента
    metrics = {
        'centroid': {'train_loss': 0, 'correct': 0, 'total': 0, 'top1_agreement': {f"teacher_{i}": 0 for i in range(len(teachers))}},
        'best': {'train_loss': 0, 'correct': 0, 'total': 0, 'top1_agreement': {f"teacher_{i}": 0 for i in range(len(teachers))}},
        'median': {'train_loss': 0, 'correct': 0, 'total': 0, 'top1_agreement': {f"teacher_{i}": 0 for i in range(len(teachers))}},
        'camkd': {'train_loss': 0, 'correct': 0, 'total': 0, 'top1_agreement': {f"teacher_{i}": 0 for i in range(len(teachers))}, 'teacher_weights': [0.0] * len(teachers), 'num_batches': 0}
    }
    
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    
    # ГЛАВНЫЙ ЦИКЛ: один проход по батчам для ВСЕ 4 студентов
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # =====================================================================
        # Вычисляем ОДИН РАЗ предсказания от ВСЕ учителей на этом батче
        # =====================================================================
        with torch.no_grad():
            logit_t_list = []
            feat_t_list = []
            
            for teacher_model, _ in teachers:
                feat_t, logit_t = teacher_model(data, is_feat=True)
                logit_t_list.append(logit_t)
                feat_t_list.append(feat_t)
            
            # Вычисляем loss для каждого учителя (для Best/Median)
            teacher_losses_batch = []
            for logit_t in logit_t_list:
                loss_t = criterion_ce(logit_t, labels)  # (batch_size,)
                teacher_losses_batch.append(loss_t.mean().item())
        
        # =====================================================================
        # МЕТОД 1: CENTROID
        # =====================================================================
        
        logit_s_centroid = students['centroid'](data)
        loss_cls_centroid = criterion_cls(logit_s_centroid, labels)
        
        with torch.no_grad():
            avg_logit_t = sum(logit_t_list) / len(logit_t_list)
        
        loss_kd_centroid = criterion_kd(logit_s_centroid, avg_logit_t)
        loss_centroid = loss_cls_centroid + config['alpha'] * loss_kd_centroid
        
        optimizers['centroid'].zero_grad()
        loss_centroid.backward()
        optimizers['centroid'].step()
        
        metrics['centroid']['train_loss'] += loss_centroid.item()
        _, predicted_centroid = logit_s_centroid.max(1)
        metrics['centroid']['correct'] += predicted_centroid.eq(labels).sum().item()
        metrics['centroid']['total'] += labels.size(0)
        
        for i, logit_t in enumerate(logit_t_list):
            teacher_preds = torch.argmax(logit_t, dim=1)
            agreement = predicted_centroid.eq(teacher_preds).sum().item()
            metrics['centroid']['top1_agreement'][f'teacher_{i}'] += agreement
        
        # =====================================================================
        # МЕТОД 2: BEST TEACHER (динамический выбор)
        # =====================================================================
        
        logit_s_best = students['best'](data)
        loss_cls_best = criterion_cls(logit_s_best, labels)
        
        best_idx = sorted(range(len(teacher_losses_batch)), 
                         key=lambda i: teacher_losses_batch[i])[0]
        best_logit_t = logit_t_list[best_idx]
        
        loss_kd_best = criterion_kd(logit_s_best, best_logit_t)
        loss_best = loss_cls_best + config['alpha'] * loss_kd_best
        
        optimizers['best'].zero_grad()
        loss_best.backward()
        optimizers['best'].step()
        
        metrics['best']['train_loss'] += loss_best.item()
        _, predicted_best = logit_s_best.max(1)
        metrics['best']['correct'] += predicted_best.eq(labels).sum().item()
        metrics['best']['total'] += labels.size(0)
        
        for i, logit_t in enumerate(logit_t_list):
            teacher_preds = torch.argmax(logit_t, dim=1)
            agreement = predicted_best.eq(teacher_preds).sum().item()
            metrics['best']['top1_agreement'][f'teacher_{i}'] += agreement
        
        # =====================================================================
        # МЕТОД 3: MEDIAN TEACHER
        # =====================================================================
        
        logit_s_median = students['median'](data)
        loss_cls_median = criterion_cls(logit_s_median, labels)
        
        sorted_indices = sorted(range(len(teacher_losses_batch)), 
                               key=lambda i: teacher_losses_batch[i])
        median_idx = sorted_indices[len(sorted_indices) // 2]
        median_logit_t = logit_t_list[median_idx]
        
        loss_kd_median = criterion_kd(logit_s_median, median_logit_t)
        loss_median = loss_cls_median + config['alpha'] * loss_kd_median
        
        optimizers['median'].zero_grad()
        loss_median.backward()
        optimizers['median'].step()
        
        metrics['median']['train_loss'] += loss_median.item()
        _, predicted_median = logit_s_median.max(1)
        metrics['median']['correct'] += predicted_median.eq(labels).sum().item()
        metrics['median']['total'] += labels.size(0)
        
        for i, logit_t in enumerate(logit_t_list):
            teacher_preds = torch.argmax(logit_t, dim=1)
            agreement = predicted_median.eq(teacher_preds).sum().item()
            metrics['median']['top1_agreement'][f'teacher_{i}'] += agreement
        
        # =====================================================================
        # МЕТОД 4: CAMKD (адаптивные веса + feature distillation ПРАВИЛЬНО)
        # =====================================================================
        
        feat_s_camkd, logit_s_camkd = students['camkd'](data, is_feat=True)
        loss_cls_camkd = criterion_cls(logit_s_camkd, labels)
        
        with torch.no_grad():
            avg_logit_t_camkd = sum(logit_t_list) / len(logit_t_list)
        
        loss_kd_camkd = criterion_kd(logit_s_camkd, avg_logit_t_camkd)
        
        # ===== Feature distillation с ConvReg =====
        # Трансформируем student features через ConvReg для каждого учителя
        trans_feat_s_list = []
        mid_feat_t_list = []
        
        hint_idx = config['hint_layer']
        if hint_idx < 0:
            hint_idx = len(feat_s_camkd) + hint_idx
        
        for feat_t, regress in zip(feat_t_list, regress_lists['camkd']):
            # Получаем features на нужном слое
            s_feat_layer = feat_s_camkd[hint_idx]
            t_feat_layer = feat_t[hint_idx]
            
            # Трансформируем student features через ConvReg
            trans_feat_s = regress(s_feat_layer)
            
            trans_feat_s_list.append(trans_feat_s)
            mid_feat_t_list.append(t_feat_layer)
        
        # Вычисляем CAMKD loss (с feature distillation)
        loss_feat_camkd, teacher_weights_camkd = criterion_camkd(
            trans_feat_s_list, mid_feat_t_list, logit_t_list, labels
        )
        
        # ВАЖНО: teacher_weights_camkd - это список ЧИСЕЛ (float), не тензоров!
        # Убеждаемся что это обычные Python float
        if isinstance(teacher_weights_camkd, (list, tuple)):
            # Это уже список - проверим что в нём именно float, а не тензоры
            teacher_weights_camkd = [w.item() if isinstance(w, torch.Tensor) else w for w in teacher_weights_camkd]
        else:
            # Если это один тензор или число - преобразуем в список
            teacher_weights_camkd = [1.0 / len(logit_t_list)] * len(logit_t_list)
        
        loss_camkd = loss_cls_camkd + config['alpha'] * loss_kd_camkd + config.get('beta', 1.0) * loss_feat_camkd
        
        optimizers['camkd'].zero_grad()
        loss_camkd.backward()
        optimizers['camkd'].step()
        
        metrics['camkd']['train_loss'] += loss_camkd.item()
        _, predicted_camkd = logit_s_camkd.max(1)
        metrics['camkd']['correct'] += predicted_camkd.eq(labels).sum().item()
        metrics['camkd']['total'] += labels.size(0)
        
        for i, logit_t in enumerate(logit_t_list):
            teacher_preds = torch.argmax(logit_t, dim=1)
            agreement = predicted_camkd.eq(teacher_preds).sum().item()
            metrics['camkd']['top1_agreement'][f'teacher_{i}'] += agreement
        
        # ИСПРАВЛЕНИЕ: убеждаемся что weights - это float перед .item()
        for i, w in enumerate(teacher_weights_camkd):
            if isinstance(w, torch.Tensor):
                metrics['camkd']['teacher_weights'][i] += w.item()
            else:
                metrics['camkd']['teacher_weights'][i] += float(w)
        
        metrics['camkd']['num_batches'] += 1
        
        # Вывод прогресса
        if batch_idx % 100 == 0:
            acc_centroid = 100. * metrics['centroid']['correct'] / max(metrics['centroid']['total'], 1)
            acc_best = 100. * metrics['best']['correct'] / max(metrics['best']['total'], 1)
            acc_median = 100. * metrics['median']['correct'] / max(metrics['median']['total'], 1)
            acc_camkd = 100. * metrics['camkd']['correct'] / max(metrics['camkd']['total'], 1)
            
            print(f"E{epoch+1} B{batch_idx}/{len(train_loader)} | "
                  f"C:{acc_centroid:.1f}% B:{acc_best:.1f}% M:{acc_median:.1f}% K:{acc_camkd:.1f}%")
    
    # Финальные метрики эпохи
    results = {}
    for method_name, metric in metrics.items():
        if metric['total'] > 0:
            train_acc = 100. * metric['correct'] / metric['total']
            avg_loss = metric['train_loss'] / len(train_loader)
            
            top1_agreement = {k: (v / metric['total'] * 100) for k, v in metric['top1_agreement'].items()}
            
            if method_name == 'camkd':
                # Усредняем weights по количеству батчей
                teacher_weights = [w / metric['num_batches'] for w in metric['teacher_weights']]
                results[method_name] = (train_acc, avg_loss, teacher_weights, top1_agreement)
            else:
                results[method_name] = (train_acc, avg_loss, top1_agreement)
    
    return results


def train_4methods_parallel(config):
    """Главная функция: обучение 4 студентов ПАРАЛЛЕЛЬНО"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    print(f"Датасет: {config['dataset']}")
    
    # Data loaders
    train_loader, test_loader, fidelity_loader = get_dataloaders(config['dataset'], config['batch_size'])
    print(f"Обучающая выборка: {len(train_loader.dataset)}")
    print(f"Тестовая выборка: {len(test_loader.dataset)}")
    
    # Teachers
    teacher_names_config = config['teacher_names']
    teachers = load_teachers(teacher_names_config, config['teacher_dir'], config['dataset'], device)
    
    if teachers is None:
        print("\n✗ Не удалось загрузить учителей!")
        return
    
    print(f"✓ Загружено учителей: {len(teachers)}")
    
    # Create save directory
    os.makedirs(f'{config["save_dir"]}/{config["dataset"].lower()}', exist_ok=True)
    
    # ========================================
    # Создаём 4 студентов и оптимайзеры
    # ========================================
    
    students = {}
    optimizers = {}
    schedulers = {}
    loggers = {}
    
    print("\n" + "="*70)
    print("Инициализация 4 студентов")
    print("="*70)
    
    for method_name in ['centroid', 'best', 'median', 'camkd']:
        student = student_model_dict['student'](num_classes=10).to(device)
        students[method_name] = student
        
        print(f"{method_name:10s}: {count_parameters(student):,} параметров")
        
        optimizer = optim.Adam(student.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        optimizers[method_name] = optimizer
        schedulers[method_name] = scheduler
        loggers[method_name] = MetricsLogger()
    
    # Setup для CAMKD (regress layers)
    regress_lists = {'centroid': [], 'best': [], 'median': [], 'camkd': []}
    
    if config['dataset'] == 'CIFAR10':
        img_size = 32
        channels = 3
    else:
        img_size = 28
        channels = 1
    
    print("\nSetup для CAMKD (regress layers)")
    dummy_input = torch.randn(2, channels, img_size, img_size).to(device)
    
    with torch.no_grad():
        output = students['camkd'](dummy_input, is_feat=True)
        feat_s, _ = output if isinstance(output, tuple) else (output, None)
        if not isinstance(feat_s, (list, tuple)):
            feat_s = [feat_s]
        
        hint_idx = config['hint_layer']
        if hint_idx < 0:
            hint_idx = len(feat_s) + hint_idx
        
        print(f"  Используем layer {hint_idx} (из {len(feat_s)})")
        
        for i, (teacher_model, teacher_name) in enumerate(teachers):
            output = teacher_model(dummy_input, is_feat=True)
            feat_t, _ = output if isinstance(output, tuple) else (output, None)
            if not isinstance(feat_t, (list, tuple)):
                feat_t = [feat_t]
            
            s_shape = list(feat_s[hint_idx].shape)
            t_shape = list(feat_t[hint_idx].shape)
            
            print(f"    {teacher_name}: s_shape={s_shape}, t_shape={t_shape}")
            
            regress = ConvReg(s_shape, t_shape).to(device)
            regress_lists['camkd'].append(regress)
    
    # Добавляем параметры regress в оптимайзер CAMKD
    camkd_params = list(students['camkd'].parameters()) + \
                   [p for regress in regress_lists['camkd'] for p in regress.parameters()]
    optimizers['camkd'] = optim.Adam(camkd_params, lr=config['lr'])
    schedulers['camkd'] = optim.lr_scheduler.StepLR(optimizers['camkd'], step_size=5, gamma=0.5)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillKL(config['temperature'])
    criterion_camkd = CAMKD()
    
    # ========================================
    # ОБУЧЕНИЕ: один цикл для всех 4 студентов
    # ========================================
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ: 4 студента параллельно на одинаковых батчах")
    print("="*70)
    
    best_accs = {'centroid': 0, 'best': 0, 'median': 0, 'camkd': 0}
    
    for epoch in range(config['epochs']):
        print(f"\nЭпоха {epoch+1}/{config['epochs']}")
        
        # Обучение
        train_results = train_epoch_parallel_4methods(
            epoch, students, teachers, regress_lists, train_loader,
            criterion_cls, criterion_kd, criterion_camkd,
            optimizers, device, config
        )
        
        # Тестирование каждого студента
        print(f"\nТестирование...")
        test_results = {}
        
        for method_name in ['centroid', 'best', 'median', 'camkd']:
            test_acc, test_loss = evaluate_model(students[method_name], test_loader, 
                                                 criterion_cls, device)
            test_results[method_name] = (test_acc, test_loss)
            
            print(f"  {method_name:10s}: Test Acc = {test_acc:.2f}%")
        
        # Вычисление fidelity метрик (периодически)
        if (epoch + 1) % config.get('fidelity_freq', 1) == 0:
            print(f"  Вычисление fidelity метрик...")
            
            for method_name in ['centroid', 'best', 'median', 'camkd']:
                teacher_models = [t[0] for t in teachers]
                fidelity_metrics = compute_fidelity_metrics(
                    students[method_name], teacher_models, fidelity_loader, 
                    device, config['temperature']
                )
                
                loggers[method_name].log(
                    epoch=epoch+1,
                    individual_fidelity=fidelity_metrics['individual_fidelity'],
                    centroid_fidelity=fidelity_metrics['centroid_fidelity'],
                    teacher_diversity=fidelity_metrics['teacher_diversity'],
                    teacher_pairwise_div=fidelity_metrics['teacher_pairwise_div'],
                    top1_agreement=fidelity_metrics['top1_agreement']
                )
        
        # Логирование метрик обучения
        for method_name in ['centroid', 'best', 'median', 'camkd']:
            train_acc, train_loss, *extra = train_results[method_name]
            test_acc, test_loss = test_results[method_name]
            
            if method_name == 'camkd':
                teacher_weights, top1_agreement = extra
                loggers[method_name].log(
                    epoch=epoch+1,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    test_acc=test_acc,
                    test_loss=test_loss,
                    teacher_weights=teacher_weights
                )
            else:
                top1_agreement = extra[0]
                loggers[method_name].log(
                    epoch=epoch+1,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    test_acc=test_acc,
                    test_loss=test_loss,
                    teacher_weights=[1.0/len(teachers)] * len(teachers)
                )
        
        # Сохранение лучших моделей
        for method_name in ['centroid', 'best', 'median', 'camkd']:
            test_acc = test_results[method_name][0]
            if test_acc > best_accs[method_name]:
                best_accs[method_name] = test_acc
                save_path = f'{config["save_dir"]}/{config["dataset"].lower()}/student_{method_name}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': students[method_name].state_dict(),
                    'test_acc': test_acc
                }, save_path)
                print(f"  ✓ Сохранена {method_name}: {test_acc:.2f}%")
        
        # Learning rate scheduler
        for scheduler in schedulers.values():
            scheduler.step()
    
    # ========================================
    # ИТОГИ И СОХРАНЕНИЕ МЕТРИК
    # ========================================
    
    print(f"\n{'='*70}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*70}")
    
    for method_name in ['centroid', 'best', 'median', 'camkd']:
        print(f"  {method_name:10s}: {best_accs[method_name]:.2f}%")
    
    print(f"{'='*70}\n")
    
    # Сохраняем метрики
    for method_name in ['centroid', 'best', 'median', 'camkd']:
        logger = loggers[method_name]
        metrics_path = f'{config["save_dir"]}/{config["dataset"].lower()}/metrics_{method_name}.json'
        logger.save(metrics_path)
        print(f"✓ Метрики сохранены: {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Обучение 4 студентов параллельно')
    parser.add_argument('--config', type=str, required=True,
                       help='Путь к конфигурационному файлу YAML')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_4methods_parallel(config)
