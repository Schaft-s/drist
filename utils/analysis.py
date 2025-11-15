#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ и визуализация результатов Knowledge Distillation
Сравнение 4 методов: Centroid, Best, Median, CAMKD
ИСПРАВЛЕНО: добавлены все импорты
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_metrics(filepath):
    """Загружает метрики из JSON"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Файл не найден: {filepath}")
        return None


def print_comparison_statistics_4methods(metrics_dict, method_names):
    """
    Выводит текстовую статистику сравнения для 4 методов
    
    Args:
        metrics_dict: {'centroid': metrics, 'best': metrics, ...}
        method_names: ['Centroid', 'Best', 'Median', 'CAMKD']
    """
    
    print("\n" + "="*80)
    print("СТАТИСТИКА СРАВНЕНИЯ: 4 МЕТОДА ДИСТИЛЛЯЦИИ")
    print("="*80)
    
    # Test Accuracy
    print(f"\nTest Accuracy (Final):")
    for method in metrics_dict:
        acc = metrics_dict[method]['test_acc'][-1]
        print(f"  {method:12s}: {acc:.2f}%")
    
    # Найти лучший
    best_method = max(metrics_dict.keys(), key=lambda m: metrics_dict[m]['test_acc'][-1])
    print(f"  → Лучший: {best_method}")
    
    # Test Loss
    print(f"\nTest Loss (Final):")
    for method in metrics_dict:
        loss = metrics_dict[method]['test_loss'][-1]
        print(f"  {method:12s}: {loss:.4f}")
    
    # Centroid Fidelity (ГЛАВНАЯ МЕТРИКА!)
    print(f"\n⭐ Centroid Fidelity (Final) - ГЛАВНАЯ МЕТРИКА:")
    for method in metrics_dict:
        centroid_fid = metrics_dict[method].get('centroid_fidelity', [])
        if centroid_fid:
            print(f"  {method:12s}: {centroid_fid[-1]:.4f}")
    
    # Найти лучший (меньше = лучше)
    best_centroid = min(metrics_dict.keys(), 
                       key=lambda m: metrics_dict[m].get('centroid_fidelity', [999])[-1] if metrics_dict[m].get('centroid_fidelity') else 999)
    print(f"  → Лучший (ближе к центроиду): {best_centroid}")
    
    # Individual Fidelity (Average)
    print(f"\nAverage Individual Fidelity (Final):")
    for method in metrics_dict:
        individual_fid = metrics_dict[method].get('individual_fidelity', {})
        if individual_fid and isinstance(individual_fid, dict):
            values = [individual_fid[k][-1] for k in individual_fid if isinstance(individual_fid[k], list) and len(individual_fid[k]) > 0]
            if values:
                avg_fid = np.mean(values)
                print(f"  {method:12s}: {avg_fid:.4f}")
    
    # Top-1 Agreement (Average)
    print(f"\nAverage Top-1 Agreement (Final %):")
    for method in metrics_dict:
        top1_agr = metrics_dict[method].get('top1_agreement', {})
        if top1_agr and isinstance(top1_agr, dict):
            values = [top1_agr[k][-1] for k in top1_agr if isinstance(top1_agr[k], list) and len(top1_agr[k]) > 0]
            if values:
                avg_agr = np.mean(values)
                print(f"  {method:12s}: {avg_agr:.1f}%")
    
    print("="*80 + "\n")


def compare_4methods_plot(metrics_dict, teacher_names, save_dir):
    """
    Строит сравнительные графики для 4 методов
    
    Args:
        metrics_dict: {'centroid': metrics, 'best': metrics, ...}
        teacher_names: список имён учителей
        save_dir: папка для сохранения
    """
    
    methods = list(metrics_dict.keys())
    colors = {'centroid': 'blue', 'best': 'red', 'median': 'green', 'camkd': 'purple'}
    markers = {'centroid': 'o', 'best': 's', 'median': '^', 'camkd': 'D'}
    
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Test Accuracy Comparison
    ax1 = plt.subplot(3, 4, 1)
    for method in methods:
        epochs = list(range(1, len(metrics_dict[method]['test_acc']) + 1))
        ax1.plot(epochs, metrics_dict[method]['test_acc'], 
                color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
                label=method.capitalize(), linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('Test Accuracy: 4 Methods', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Test Loss Comparison
    ax2 = plt.subplot(3, 4, 2)
    for method in methods:
        epochs = list(range(1, len(metrics_dict[method]['test_loss']) + 1))
        ax2.plot(epochs, metrics_dict[method]['test_loss'], 
                color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
                label=method.capitalize(), linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Test Loss', fontsize=11)
    ax2.set_title('Test Loss: 4 Methods', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Centroid Fidelity ⭐ ГЛАВНЫЙ ГРАФИК
    ax3 = plt.subplot(3, 4, 3)
    for method in methods:
        centroid_fid = metrics_dict[method].get('centroid_fidelity', [])
        if centroid_fid:
            fid_epochs = list(range(1, len(centroid_fid) + 1))
            ax3.plot(fid_epochs, centroid_fid, 
                    color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
                    label=method.capitalize(), linewidth=2, markersize=5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('KL Divergence', fontsize=11)
    ax3.set_title('⭐ Centroid Fidelity (ключевая)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Accuracy Bar Chart
    ax4 = plt.subplot(3, 4, 4)
    final_accs = [metrics_dict[m]['test_acc'][-1] for m in methods]
    bars = ax4.bar(methods, final_accs, color=[colors.get(m, 'gray') for m in methods])
    ax4.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax4.set_title('Final Test Accuracy', fontsize=12, fontweight='bold')
    ax4.set_ylim([min(final_accs) - 1, max(final_accs) + 1])
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5-8. Individual Fidelity для первых 4 учителей
    for idx in range(min(4, 5)):  # Максимум 4 графика
        ax = plt.subplot(3, 4, 5 + idx)
        for method in methods:
            individual_fid = metrics_dict[method].get('individual_fidelity', {})
            key = f"teacher_{idx}"
            if individual_fid and key in individual_fid:
                fid_data = individual_fid[key]
                if isinstance(fid_data, list) and len(fid_data) > 0:
                    fid_epochs = list(range(1, len(fid_data) + 1))
                    ax.plot(fid_epochs, fid_data, 
                           color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
                           label=method.capitalize(), linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('KL Divergence', fontsize=11)
        teacher_name = teacher_names[idx] if idx < len(teacher_names) else f"Teacher {idx}"
        ax.set_title(f'Individual Fidelity: {teacher_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 9. Average Top-1 Agreement
    ax9 = plt.subplot(3, 4, 9)
    for method in methods:
        top1_agr = metrics_dict[method].get('top1_agreement', {})
        if top1_agr and isinstance(top1_agr, dict):
            # Average across all teachers
            num_epochs = len(list(top1_agr.values())[0]) if top1_agr.values() else 0
            if num_epochs > 0:
                avg_agreements = []
                for epoch_idx in range(num_epochs):
                    epoch_agr = [top1_agr[k][epoch_idx] for k in top1_agr 
                               if isinstance(top1_agr[k], list) and epoch_idx < len(top1_agr[k])]
                    if epoch_agr:
                        avg_agreements.append(np.mean(epoch_agr))
                
                if avg_agreements:
                    agr_epochs = list(range(1, len(avg_agreements) + 1))
                    ax9.plot(agr_epochs, avg_agreements, 
                            color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
                            label=method.capitalize(), linewidth=2, markersize=5)
    ax9.set_xlabel('Epoch', fontsize=11)
    ax9.set_ylabel('Top-1 Agreement (%)', fontsize=11)
    ax9.set_title('Average Top-1 Agreement', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([0, 105])
    
    # 10. Final Centroid Fidelity Bar Chart
    ax10 = plt.subplot(3, 4, 10)
    final_centroid = [metrics_dict[m].get('centroid_fidelity', [0])[-1] 
                     if metrics_dict[m].get('centroid_fidelity') else 0 
                     for m in methods]
    bars = ax10.bar(methods, final_centroid, color=[colors.get(m, 'gray') for m in methods])
    ax10.set_ylabel('KL Divergence', fontsize=11)
    ax10.set_title('⭐ Final Centroid Fidelity', fontsize=12, fontweight='bold')
    for bar, fid in zip(bars, final_centroid):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                f'{fid:.4f}', ha='center', va='bottom', fontsize=10)
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. Teacher Weights для CAMKD
    ax11 = plt.subplot(3, 4, 11)
    if 'camkd' in metrics_dict:
        teacher_weights = metrics_dict['camkd'].get('teacher_weights', {})
        if teacher_weights and isinstance(teacher_weights, dict):
            weight_epochs = list(range(1, len(list(teacher_weights.values())[0]) + 1))
            for i, (key, weights) in enumerate(teacher_weights.items()):
                teacher_name = teacher_names[i] if i < len(teacher_names) else key
                ax11.plot(weight_epochs, weights, marker='o', label=teacher_name, linewidth=2)
        ax11.set_xlabel('Epoch', fontsize=11)
        ax11.set_ylabel('Weight', fontsize=11)
        ax11.set_title('CAMKD: Adaptive Teacher Weights', fontsize=12, fontweight='bold')
        ax11.legend(fontsize=9)
        ax11.grid(True, alpha=0.3)
    
    # 12. Overfitting Gap Comparison
    ax12 = plt.subplot(3, 4, 12)
    for method in methods:
        train_acc = metrics_dict[method]['train_acc']
        test_acc = metrics_dict[method]['test_acc']
        gap = [tr - te for tr, te in zip(train_acc, test_acc)]
        epochs = list(range(1, len(gap) + 1))
        ax12.plot(epochs, gap, 
                 color=colors.get(method, 'gray'), marker=markers.get(method, 'o'),
                 label=method.capitalize(), linewidth=2, markersize=5)
    ax12.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax12.set_xlabel('Epoch', fontsize=11)
    ax12.set_ylabel('Gap (%)', fontsize=11)
    ax12.set_title('Overfitting Gap', fontsize=12, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f'{save_dir}/comparison_4methods.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Графики сохранены: {save_path} (12 графиков)")
    plt.close()


def compare_4methods(dataset, save_dir='./results'):
    """
    Главная функция для сравнения 4 методов
    
    Загружает метрики для: Centroid, Best, Median, CAMKD
    """
    
    dataset_dir = f'{save_dir}/{dataset.lower()}'
    
    print(f"\n🔍 Загрузка метрик из: {dataset_dir}/")
    
    # Load metrics for all 4 methods
    methods = ['centroid', 'best', 'median', 'camkd']
    metrics_dict = {}
    
    for method in methods:
        metrics_path = f'{dataset_dir}/metrics_{method}.json'
        metrics = load_metrics(metrics_path)
        if metrics is None:
            print(f"\n⚠️ Метрики для {method} не найдены!")
            print(f"Ожидался файл: {metrics_path}")
        else:
            metrics_dict[method] = metrics
    
    if len(metrics_dict) == 0:
        print("\n✗ Не удалось загрузить ни одной метрики!")
        print("Сначала запустите обучение:")
        print("  python train_student_4methods.py --config ...")
        return False
    
    # Infer teacher names from first method
    first_method = list(metrics_dict.values())[0]
    num_teachers = len(first_method.get('individual_fidelity', {}))
    teacher_names = [f'Teacher {i+1}' for i in range(num_teachers)]
    
    # Print statistics
    print_comparison_statistics_4methods(metrics_dict, teacher_names)
    
    # Plot comparison
    compare_4methods_plot(metrics_dict, teacher_names, dataset_dir)
    
    print(f"✓ Анализ завершён!")
    print(f"✓ Результаты сохранены в: {dataset_dir}/")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Анализ результатов: 4 метода дистилляции')
    parser.add_argument('--dataset', type=str, default='FashionMNIST',
                       choices=['MNIST', 'FashionMNIST', 'CIFAR10'])
    parser.add_argument('--save_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    success = compare_4methods(args.dataset, args.save_dir)
    
    if not success:
        exit(1)
