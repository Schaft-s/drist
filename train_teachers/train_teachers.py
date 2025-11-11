#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение учителей на выбранном датасете
Поддержка 5 различных архитектур
Использует конфигурационные файлы YAML
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import yaml
import sys

# Добавляем родительскую директорию в путь для импорта
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from train_teachers.teacher_models import teacher_model_dict, count_parameters


def load_config(config_path):
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser('Обучение учителей')
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

    return train_loader, test_loader


def train_model(model, model_name, train_loader, test_loader, device, epochs, lr, save_path):
    """Обучает модель"""
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    print(f"\n{'='*60}")
    print(f"Обучение: {model_name}")
    print(f"Параметров: {count_parameters(model):,}")
    print(f"{'='*60}")

    best_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'  E{epoch+1}/{epochs} B{batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')

        train_acc = 100. * correct / total

        # Testing
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_acc = 100. * correct / total

        print(f'\n  Epoch {epoch+1}: Train={train_acc:.2f}% | Test={test_acc:.2f}%\n')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
            }, save_path)
            print(f'  ✓ Сохранена лучшая модель: {test_acc:.2f}%')

        scheduler.step()

    print(f'\n{model_name} - Итоговая точность: {best_acc:.2f}%\n')
    return best_acc


if __name__ == '__main__':
    args = parse_args()

    # Загружаем конфигурацию
    config = load_config(args.config)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f'{config["save_dir"]}/{config["dataset"].lower()}'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Устройство: {device}")
    print(f"Датасет: {config['dataset']}")
    print(f"Конфигурация: {args.config}")

    # Data loaders
    train_loader, test_loader = get_dataloaders(config['dataset'], config['batch_size'])

    print(f"\nОбучающая выборка: {len(train_loader.dataset)}")
    print(f"Тестовая выборка: {len(test_loader.dataset)}")

    # Train teachers
    teacher_names = config['teachers']
    num_classes = 10

    results = {}
    for name in teacher_names:
        if name not in teacher_model_dict:
            print(f"\n⚠️ Неизвестная модель: {name}")
            print(f"Доступные: {list(teacher_model_dict.keys())}")
            continue

        model = teacher_model_dict[name](num_classes=num_classes)
        save_path = f'{save_dir}/{name}.pth'

        acc = train_model(model, name, train_loader, test_loader, 
                         device, config['epochs'], config['lr'], save_path)
        results[name] = acc

    print(f"\n{'='*60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name}: {acc:.2f}%")
    print(f"{'='*60}\n")

