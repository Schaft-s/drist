#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подробный анализ результатов Knowledge Distillation
Сравнение 4 методов: Centroid, Best, Median, CAMKD

Строит 12 сравнительных графиков и выводит статистику
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import os


def load_metrics(filepath):
    """Загружает метрики из JSON"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Файл не найден: {filepath}")
        return None


def print_section(title):
    """Печатает заголовок секции"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_comparison_statistics(metrics_dict):
    """
    Выводит детальную текстовую статистику для всех методов
    """
    
    print_section("СТАТИСТИКА СРАВНЕНИЯ: 4 МЕТОДА ДИСТИЛЛЯЦИИ")
    
    methods = list(metrics_dict.keys())
    
    # =====================================================================
    # 1. TEST ACCURACY
    # =====================================================================
    print("\n📊 TEST ACCURACY (Final):")
    print("-" * 50)
    
    final_accs = {}
    for method in methods:
        acc = metrics_dict[method]['test_acc'][-1]
        final_accs[method] = acc
        status = "★" if acc == max([metrics_dict[m]['test_acc'][-1] for m in methods]) else " "
        print(f"  {status} {method:12s}: {acc:7.2f}%")
    
    best_acc_method = max(methods, key=lambda m: final_accs[m])
    print(f"\n  → Лучший: {best_acc_method} ({final_accs[best_acc_method]:.2f}%)")
    
    # =====================================================================
    # 2. TEST LOSS
    # =====================================================================
    print("\n📊 TEST LOSS (Final):")
    print("-" * 50)
    
    final_losses = {}
    for method in methods:
        loss = metrics_dict[method]['test_loss'][-1]
        final_losses[method] = loss
        status = "★" if loss == min([metrics_dict[m]['test_loss'][-1] for m in methods]) else " "
        print(f"  {status} {method:12s}: {loss:9.4f}")
    
    best_loss_method = min(methods, key=lambda m: final_losses[m])
    print(f"\n  → Лучший: {best_loss_method} ({final_losses[best_loss_method]:.4f})")
    
    # =====================================================================
    # 3. CENTROID FIDELITY (ГЛАВНАЯ МЕТРИКА!)
    # =====================================================================
    print("\n⭐ CENTROID FIDELITY (Final) - ГЛАВНАЯ МЕТРИКА:")
    print("-" * 50)
    print("    (меньше = ближе к центроиду = лучше)")
    
    final_centroid_fid = {}
    for method in methods:
        centroid_fid = metrics_dict[method].get('centroid_fidelity', [])
        if centroid_fid:
            fid = centroid_fid[-1]
            final_centroid_fid[method] = fid
            status = "★" if fid == min([metrics_dict[m].get('centroid_fidelity', [999])[-1] 
                                       for m in methods if metrics_dict[m].get('centroid_fidelity')]) else " "
            print(f"  {status} {method:12s}: {fid:9.4f}")
    
    best_centroid_method = min(methods, key=lambda m: final_centroid_fid.get(m, 999))
    print(f"\n  → Лучший: {best_centroid_method} ({final_centroid_fid[best_centroid_method]:.4f})")
    print(f"\n  ✅ ГИПОТЕЗА: Centroid должен быть НИЖЕ Best")
    
    if 'centroid' in final_centroid_fid and 'best' in final_centroid_fid:
        if final_centroid_fid['centroid'] < final_centroid_fid['best']:
            print(f"     ✓ ПОДТВЕРЖДЕНА! Centroid ({final_centroid_fid['centroid']:.4f}) < Best ({final_centroid_fid['best']:.4f})")
        else:
            print(f"     ✗ НЕ подтверждена. Centroid ({final_centroid_fid['centroid']:.4f}) >= Best ({final_centroid_fid['best']:.4f})")
    
    # =====================================================================
    # 4. INDIVIDUAL FIDELITY (Average)
    # =====================================================================
    print("\n📊 AVERAGE INDIVIDUAL FIDELITY (Final):")
    print("-" * 50)
    print("    (средняя fidelity по всем учителям)")
    
    avg_individual_fid = {}
    for method in methods:
        individual_fid = metrics_dict[method].get('individual_fidelity', {})
        if individual_fid and isinstance(individual_fid, dict):
            values = [individual_fid[k][-1] for k in individual_fid 
                     if isinstance(individual_fid[k], list) and len(individual_fid[k]) > 0]
            if values:
                avg_fid = np.mean(values)
                avg_individual_fid[method] = avg_fid
                status = "★" if avg_fid == min([np.mean([individual_fid[k][-1] for k in individual_fid 
                                                         if isinstance(individual_fid[k], list) and len(individual_fid[k]) > 0])
                                               for method in methods 
                                               if metrics_dict[method].get('individual_fidelity')]) else " "
                print(f"  {status} {method:12s}: {avg_fid:9.4f}")
    
    # =====================================================================
    # 5. TOP-1 AGREEMENT (Average)
    # =====================================================================
    print("\n📊 AVERAGE TOP-1 AGREEMENT (Final, %):")
    print("-" * 50)
    print("    (средний процент согласия с учителями)")
    
    avg_top1_agr = {}
    for method in methods:
        top1_agr = metrics_dict[method].get('top1_agreement', {})
        if top1_agr and isinstance(top1_agr, dict):
            values = [top1_agr[k][-1] for k in top1_agr 
                     if isinstance(top1_agr[k], list) and len(top1_agr[k]) > 0]
            if values:
                avg_agr = np.mean(values)
                avg_top1_agr[method] = avg_agr
                status = "★" if avg_agr == max([np.mean([top1_agr[k][-1] for k in top1_agr 
                                                         if isinstance(top1_agr[k], list) and len(top1_agr[k]) > 0])
                                               for method in methods 
                                               if metrics_dict[method].get('top1_agreement')]) else " "
                print(f"  {status} {method:12s}: {avg_agr:7.1f}%")
    
    # =====================================================================
    # 6. EARLY EPOCHS ANALYSIS (Epoch 1 vs Final)
    # =====================================================================
    print("\n📊 ДИНАМИКА ОБУЧЕНИЯ (Epoch 1 vs Final):")
    print("-" * 50)
    
    for method in methods:
        acc_1 = metrics_dict[method]['test_acc'][0]
        acc_final = metrics_dict[method]['test_acc'][-1]
        improvement = acc_final - acc_1
        
        print(f"  {method:12s}: {acc_1:6.2f}% → {acc_final:6.2f}% (Δ {improvement:+6.2f}%)")
    
    # =====================================================================
    # 7. TEACHER WEIGHTS для CAMKD
    # =====================================================================
    print("\n📊 TEACHER WEIGHTS (Final, только CAMKD):")
    print("-" * 50)
    
    if 'camkd' in metrics_dict:
        teacher_weights = metrics_dict['camkd'].get('teacher_weights', {})
        if teacher_weights and isinstance(teacher_weights, dict):
            print("  Адаптивные веса учителей (CAMKD):")
            total_weight = 0
            weights_list = []
            for i, (key, weights) in enumerate(sorted(teacher_weights.items())):
                if isinstance(weights, list) and len(weights) > 0:
                    final_weight = weights[-1]
                    weights_list.append((key, final_weight))
                    total_weight += final_weight
                    print(f"    {key}: {final_weight:.4f}")
            
            if total_weight > 0:
                print(f"  \n  Нормализованные (сумма = {total_weight:.4f}):")
                for key, weight in weights_list:
                    normalized = weight / total_weight
                    print(f"    {key}: {normalized:.4f} ({normalized*100:.1f}%)")
    
    # =====================================================================
    # 8. OVERFITTING GAP (Generalization)
    # =====================================================================
    print("\n📊 OVERFITTING GAP (Train-Test Difference, Final):")
    print("-" * 50)
    print("    (меньше = лучше обобщение)")
    
    for method in methods:
        train_acc = metrics_dict[method]['train_acc'][-1]
        test_acc = metrics_dict[method]['test_acc'][-1]
        gap = train_acc - test_acc
        
        print(f"  {method:12s}: {gap:6.2f}% (Train: {train_acc:.2f}%, Test: {test_acc:.2f}%)")
    
    print("=" * 80 + "\n")


def plot_comparison_12figures(metrics_dict, teacher_names, save_dir):
    """
    Строит 12 сравнительных графиков для анализа 4 методов
    """
    
    methods = list(metrics_dict.keys())
    colors = {'centroid': '#1f77b4', 'best': '#ff7f0e', 'median': '#2ca02c', 'camkd': '#d62728'}
    markers = {'centroid': 'o', 'best': 's', 'median': '^', 'camkd': 'D'}
    
    fig = plt.figure(figsize=(24, 16))
    
    # =====================================================================
    # 1. Test Accuracy по эпохам
    # =====================================================================
    ax1 = plt.subplot(3, 4, 1)
    for method in methods:
        epochs = list(range(1, len(metrics_dict[method]['test_acc']) + 1))
        ax1.plot(epochs, metrics_dict[method]['test_acc'], 
                color=colors[method], marker=markers[method],
                label=method.capitalize(), linewidth=2.5, markersize=6, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy по эпохам', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([80, 95])
    
    # =====================================================================
    # 2. Test Loss по эпохам
    # =====================================================================
    ax2 = plt.subplot(3, 4, 2)
    for method in methods:
        epochs = list(range(1, len(metrics_dict[method]['test_loss']) + 1))
        ax2.plot(epochs, metrics_dict[method]['test_loss'], 
                color=colors[method], marker=markers[method],
                label=method.capitalize(), linewidth=2.5, markersize=6, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Test Loss по эпохам', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # =====================================================================
    # 3. ⭐ Centroid Fidelity (ГЛАВНЫЙ ГРАФИК)
    # =====================================================================
    ax3 = plt.subplot(3, 4, 3)
    for method in methods:
        centroid_fid = metrics_dict[method].get('centroid_fidelity', [])
        if centroid_fid:
            fid_epochs = list(range(1, len(centroid_fid) + 1))
            ax3.plot(fid_epochs, centroid_fid, 
                    color=colors[method], marker=markers[method],
                    label=method.capitalize(), linewidth=2.5, markersize=6, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax3.set_title('⭐ Centroid Fidelity (КЛЮЧЕВОЙ)', fontsize=13, fontweight='bold', color='red')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # =====================================================================
    # 4. Final Accuracy Bar Chart
    # =====================================================================
    ax4 = plt.subplot(3, 4, 4)
    final_accs = [metrics_dict[m]['test_acc'][-1] for m in methods]
    bars = ax4.bar(methods, final_accs, color=[colors[m] for m in methods], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Final Test Accuracy', fontsize=13, fontweight='bold')
    ax4.set_ylim([85, 95])
    
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # =====================================================================
    # 5-8. Individual Fidelity для первых 4 учителей
    # =====================================================================
    for idx in range(min(4, len(teacher_names))):
        ax = plt.subplot(3, 4, 5 + idx)
        
        for method in methods:
            individual_fid = metrics_dict[method].get('individual_fidelity', {})
            key = f"teacher_{idx}"
            if individual_fid and key in individual_fid:
                fid_data = individual_fid[key]
                if isinstance(fid_data, list) and len(fid_data) > 0:
                    fid_epochs = list(range(1, len(fid_data) + 1))
                    ax.plot(fid_epochs, fid_data, 
                           color=colors[method], marker=markers[method],
                           label=method.capitalize(), linewidth=2, markersize=5, alpha=0.8)
        
        teacher_name = teacher_names[idx] if idx < len(teacher_names) else f"Teacher {idx}"
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('KL Divergence', fontsize=11, fontweight='bold')
        ax.set_title(f'Individual Fidelity:\n{teacher_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # =====================================================================
    # 9. Average Top-1 Agreement
    # =====================================================================
    ax9 = plt.subplot(3, 4, 9)
    for method in methods:
        top1_agr = metrics_dict[method].get('top1_agreement', {})
        if top1_agr and isinstance(top1_agr, dict):
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
                            color=colors[method], marker=markers[method],
                            label=method.capitalize(), linewidth=2.5, markersize=6, alpha=0.8)
    
    ax9.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Random')
    ax9.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Top-1 Agreement (%)', fontsize=12, fontweight='bold')
    ax9.set_title('Average Top-1 Agreement', fontsize=13, fontweight='bold')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([40, 100])
    
    # =====================================================================
    # 10. Final Centroid Fidelity Bar Chart
    # =====================================================================
    ax10 = plt.subplot(3, 4, 10)
    final_centroid = [metrics_dict[m].get('centroid_fidelity', [0])[-1] 
                     if metrics_dict[m].get('centroid_fidelity') else 0 
                     for m in methods]
    bars = ax10.bar(methods, final_centroid, color=[colors[m] for m in methods], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax10.set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    ax10.set_title('⭐ Final Centroid Fidelity', fontsize=13, fontweight='bold', color='red')
    
    for bar, fid in zip(bars, final_centroid):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                f'{fid:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # =====================================================================
    # 11. CAMKD Teacher Weights
    # =====================================================================
    ax11 = plt.subplot(3, 4, 11)
    if 'camkd' in metrics_dict:
        teacher_weights = metrics_dict['camkd'].get('teacher_weights', {})
        if teacher_weights and isinstance(teacher_weights, dict):
            weight_epochs = list(range(1, len(list(teacher_weights.values())[0]) + 1)) if teacher_weights.values() else []
            for i, (key, weights) in enumerate(sorted(teacher_weights.items())):
                if isinstance(weights, list) and len(weights) > 0:
                    teacher_name = teacher_names[i] if i < len(teacher_names) else key
                    ax11.plot(weight_epochs, weights, marker='o', label=teacher_name, linewidth=2.5, markersize=6)
        
        ax11.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax11.set_ylabel('Weight', fontsize=12, fontweight='bold')
        ax11.set_title('CAMKD: Adaptive Teacher Weights', fontsize=13, fontweight='bold')
        ax11.legend(fontsize=10)
        ax11.grid(True, alpha=0.3)
    
    # =====================================================================
    # 12. Overfitting Gap
    # =====================================================================
    ax12 = plt.subplot(3, 4, 12)
    for method in methods:
        train_acc = metrics_dict[method]['train_acc']
        test_acc = metrics_dict[method]['test_acc']
        gap = [tr - te for tr, te in zip(train_acc, test_acc)]
        epochs = list(range(1, len(gap) + 1))
        ax12.plot(epochs, gap, 
                 color=colors[method], marker=markers[method],
                 label=method.capitalize(), linewidth=2.5, markersize=6, alpha=0.8)
    
    ax12.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax12.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax12.set_ylabel('Overfitting Gap (%)', fontsize=12, fontweight='bold')
    ax12.set_title('Train-Test Overfitting Gap', fontsize=13, fontweight='bold')
    ax12.legend(fontsize=10)
    ax12.grid(True, alpha=0.3)
    
    plt.suptitle('Knowledge Distillation: 4 Methods Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = f'{save_dir}/comparison_4methods.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Графики сохранены: {save_path}")
    plt.close()


def analyze_4methods(dataset, save_dir='./results'):
    """Главная функция анализа"""
    
    dataset_dir = f'{save_dir}/{dataset.lower()}'
    
    print_section(f"АНАЛИЗ РЕЗУЛЬТАТОВ: {dataset}")
    print(f"\n📂 Загрузка метрик из: {dataset_dir}/\n")
    
    # Загружаем метрики
    methods = ['centroid', 'best', 'median', 'camkd']
    metrics_dict = {}
    
    for method in methods:
        metrics_path = f'{dataset_dir}/metrics_{method}.json'
        metrics = load_metrics(metrics_path)
        
        if metrics is None:
            print(f"❌ Не найдены метрики для {method}")
            print(f"   Ожидается: {metrics_path}")
        else:
            metrics_dict[method] = metrics
            print(f"✓ Загружены метрики для {method}")
    
    if len(metrics_dict) == 0:
        print("\n❌ Не удалось загрузить ни одной метрики!")
        print("Сначала запустите обучение:")
        print("  python train_student/train_student_4methods_parallel.py --config ...")
        return False
    
    print(f"\n✓ Успешно загружено {len(metrics_dict)} методов\n")
    
    # Определяем имена учителей
    first_method = list(metrics_dict.values())[0]
    num_teachers = len(first_method.get('individual_fidelity', {}))
    teacher_names = [f'T{i+1}' for i in range(num_teachers)]
    
    # Выводим статистику
    print_comparison_statistics(metrics_dict)
    
    # Строим графики
    print_section("ПОСТРОЕНИЕ ГРАФИКОВ")
    plot_comparison_12figures(metrics_dict, teacher_names, dataset_dir)
    
    print_section("✅ АНАЛИЗ ЗАВЕРШЁН")
    print(f"\n📊 Результаты сохранены в: {dataset_dir}/")
    print(f"   - comparison_4methods.png (12 графиков)")
    print("\n" + "="*80 + "\n")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Подробный анализ 4 методов дистилляции')
    parser.add_argument('--dataset', type=str, default='FashionMNIST',
                       choices=['MNIST', 'FashionMNIST', 'CIFAR10'],
                       help='Датасет для анализа')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Директория сохранения результатов')
    
    args = parser.parse_args()
    
    success = analyze_4methods(args.dataset, args.save_dir)
    
    if not success:
        exit(1)
