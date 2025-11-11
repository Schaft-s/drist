#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ и визуализация результатов Knowledge Distillation
Сравнение CAMKD vs Vanilla KD с поддержкой Top-1 Agreement
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_metrics(filepath):
    """Загружает метрики из JSON"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Файл не найден: {filepath}")
        return None


def compare_two_methods(metrics_camkd, metrics_vanilla, teacher_names, save_dir):
    """
    Создает 14 сравнительных графиков (4x4) для двух методов
    """

    fig = plt.figure(figsize=(24, 18))

    epochs_camkd = list(range(1, len(metrics_camkd['test_acc']) + 1))
    epochs_vanilla = list(range(1, len(metrics_vanilla['test_acc']) + 1))

    # ========================================================================
    # 1. Test Accuracy Comparison
    # ========================================================================
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(epochs_camkd, metrics_camkd['test_acc'], 'b-o', 
            label='CAMKD', linewidth=2, markersize=5)
    ax1.plot(epochs_vanilla, metrics_vanilla['test_acc'], 'r-s', 
            label='Vanilla KD', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('1. Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # 2. Test Loss Comparison
    # ========================================================================
    ax2 = plt.subplot(4, 4, 2)
    ax2.plot(epochs_camkd, metrics_camkd['test_loss'], 'b-o', 
            label='CAMKD', linewidth=2, markersize=5)
    ax2.plot(epochs_vanilla, metrics_vanilla['test_loss'], 'r-s', 
            label='Vanilla KD', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Test Loss', fontsize=11)
    ax2.set_title('2. Test Loss Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # 3. Accuracy Improvement (CAMKD - Vanilla)
    # ========================================================================
    ax3 = plt.subplot(4, 4, 3)
    min_len = min(len(metrics_camkd['test_acc']), len(metrics_vanilla['test_acc']))
    acc_diff = [c - v for c, v in zip(metrics_camkd['test_acc'][:min_len], 
                                      metrics_vanilla['test_acc'][:min_len])]
    ax3.plot(range(1, len(acc_diff) + 1), acc_diff, 'g-^', linewidth=2, markersize=6)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(range(1, len(acc_diff) + 1), acc_diff, 0, alpha=0.3, color='green')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy Difference (%)', fontsize=11)
    ax3.set_title('3. CAMKD - Vanilla (+green=CAMKD better)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # 4-6. Individual Fidelity
    # ========================================================================
    individual_camkd = metrics_camkd.get('individual_fidelity', {})
    individual_vanilla = metrics_vanilla.get('individual_fidelity', {})

    if individual_camkd and individual_vanilla and isinstance(individual_camkd, dict):
        # Сортируем ключи для консистентности
        keys = sorted(individual_camkd.keys())

        for plot_idx, key in enumerate(keys[:3]):  # Первые 3 учителя
            ax = plt.subplot(4, 4, 4 + plot_idx)
            camkd_vals = individual_camkd[key] if isinstance(individual_camkd[key], list) else []
            vanilla_vals = individual_vanilla.get(key, []) if isinstance(individual_vanilla.get(key), list) else []

            if camkd_vals and vanilla_vals:
                epochs_fid = list(range(1, len(camkd_vals) + 1))
                ax.plot(epochs_fid, camkd_vals, 'b-o', label='CAMKD', linewidth=2, markersize=4)
                ax.plot(epochs_fid, vanilla_vals, 'r-s', label='Vanilla', linewidth=2, markersize=4)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('KL Divergence', fontsize=10)
                ax.set_title(f'{4+plot_idx}. Individual Fidelity: {key}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

    # ========================================================================
    # 7. Centroid Fidelity
    # ========================================================================
    ax7 = plt.subplot(4, 4, 7)
    centroid_camkd = metrics_camkd.get('centroid_fidelity', [])
    centroid_vanilla = metrics_vanilla.get('centroid_fidelity', [])

    if centroid_camkd and centroid_vanilla:
        min_len = min(len(centroid_camkd), len(centroid_vanilla))
        epochs_c = list(range(1, min_len + 1))
        ax7.plot(epochs_c, centroid_camkd[:min_len], 'b-o', 
                label='CAMKD', linewidth=2, markersize=5)
        ax7.plot(epochs_c, centroid_vanilla[:min_len], 'r-s', 
                label='Vanilla', linewidth=2, markersize=5)
        ax7.set_xlabel('Epoch', fontsize=11)
        ax7.set_ylabel('KL Divergence', fontsize=11)
        ax7.set_title('7. Centroid Fidelity', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)

    # ========================================================================
    # 8. Average Individual Fidelity
    # ========================================================================
    ax8 = plt.subplot(4, 4, 8)
    if individual_camkd and individual_vanilla and isinstance(individual_camkd, dict):
        keys = sorted(individual_camkd.keys())
        min_len_fid = min(len(individual_camkd[keys[0]]) if isinstance(individual_camkd[keys[0]], list) else 0 
                         for key in keys if key in individual_camkd)

        if min_len_fid > 0:
            avg_camkd = [np.mean([individual_camkd[key][e] 
                                 for key in keys if isinstance(individual_camkd[key], list) and e < len(individual_camkd[key])]) 
                        for e in range(min_len_fid)]
            avg_vanilla = [np.mean([individual_vanilla[key][e] 
                                   for key in keys if isinstance(individual_vanilla[key], list) and e < len(individual_vanilla[key])]) 
                          for e in range(min_len_fid)]

            epochs_avg = list(range(1, len(avg_camkd) + 1))
            ax8.plot(epochs_avg, avg_camkd, 'b-o', label='CAMKD', linewidth=2, markersize=5)
            ax8.plot(epochs_avg, avg_vanilla, 'r-s', label='Vanilla', linewidth=2, markersize=5)
            ax8.set_xlabel('Epoch', fontsize=11)
            ax8.set_ylabel('Avg KL Divergence', fontsize=11)
            ax8.set_title('8. Average Individual Fidelity', fontsize=12, fontweight='bold')
            ax8.legend(fontsize=10)
            ax8.grid(True, alpha=0.3)

    # ========================================================================
    # 9. Teacher Weights (только CAMKD + справка о Vanilla)
    # ========================================================================
    ax9 = plt.subplot(4, 4, 9)
    teacher_weights_camkd = metrics_camkd.get('teacher_weights', {})

    if teacher_weights_camkd and isinstance(teacher_weights_camkd, dict):
        keys = sorted(teacher_weights_camkd.keys())
        min_len_w = min(len(teacher_weights_camkd[key]) 
                       for key in keys if isinstance(teacher_weights_camkd[key], list))

        if min_len_w > 0:
            epochs_w = list(range(1, min_len_w + 1))
            for key in keys:
                if isinstance(teacher_weights_camkd[key], list):
                    ax9.plot(epochs_w, teacher_weights_camkd[key][:min_len_w], 
                            marker='o', label=f'{key} (CAMKD)', linewidth=2)

            # Линия для равных весов (Vanilla)
            equal_weight = 1.0 / len(keys)
            ax9.axhline(y=equal_weight, color='gray', linestyle='--', 
                       linewidth=2, label='Vanilla (equal)', alpha=0.7)

            ax9.set_xlabel('Epoch', fontsize=11)
            ax9.set_ylabel('Weight', fontsize=11)
            ax9.set_title('9. Teacher Weights (CAMKD vs Vanilla)', fontsize=12, fontweight='bold')
            ax9.legend(fontsize=9)
            ax9.grid(True, alpha=0.3)

    # ========================================================================
    # 10. Final Metrics Bar Chart
    # ========================================================================
    ax10 = plt.subplot(4, 4, 10)
    metrics_to_plot = ['Accuracy', 'Loss']
    camkd_vals = [metrics_camkd['test_acc'][-1], metrics_camkd['test_loss'][-1]]
    vanilla_vals = [metrics_vanilla['test_acc'][-1], metrics_vanilla['test_loss'][-1]]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    bars1 = ax10.bar(x - width/2, camkd_vals, width, label='CAMKD', alpha=0.8, color='blue')
    bars2 = ax10.bar(x + width/2, vanilla_vals, width, label='Vanilla', alpha=0.8, color='red')

    ax10.set_ylabel('Value', fontsize=11)
    ax10.set_title('10. Final Metrics', fontsize=12, fontweight='bold')
    ax10.set_xticks(x)
    ax10.set_xticklabels(metrics_to_plot)
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # ========================================================================
    # 11. Final Individual Fidelity Bar Chart
    # ========================================================================
    ax11 = plt.subplot(4, 4, 11)
    if individual_camkd and individual_vanilla and isinstance(individual_camkd, dict):
        keys = sorted(individual_camkd.keys())[:3]  # Первые 3
        final_camkd = [individual_camkd[key][-1] if isinstance(individual_camkd[key], list) else 0 for key in keys]
        final_vanilla = [individual_vanilla.get(key, [0])[-1] if isinstance(individual_vanilla.get(key), list) else 0 for key in keys]

        x = np.arange(len(keys))
        width = 0.35

        bars1 = ax11.bar(x - width/2, final_camkd, width, label='CAMKD', alpha=0.8, color='blue')
        bars2 = ax11.bar(x + width/2, final_vanilla, width, label='Vanilla', alpha=0.8, color='red')

        ax11.set_ylabel('KL Divergence', fontsize=11)
        ax11.set_title('11. Final Individual Fidelity', fontsize=12, fontweight='bold')
        ax11.set_xticks(x)
        ax11.set_xticklabels([k.replace('teacher_', 'T') for k in keys], rotation=0)
        ax11.legend(fontsize=10)
        ax11.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # ========================================================================
    # 12. Top-1 Agreement Comparison ⭐
    # ========================================================================
    ax12 = plt.subplot(4, 4, 12)
    top1_camkd = metrics_camkd.get('top1_agreement', {})
    top1_vanilla = metrics_vanilla.get('top1_agreement', {})

    if top1_camkd and top1_vanilla and isinstance(top1_camkd, dict):
        keys = sorted(top1_camkd.keys())
        min_len_top1 = min(len(top1_camkd[key]) for key in keys if isinstance(top1_camkd[key], list))

        if min_len_top1 > 0:
            epochs_top1 = list(range(1, min_len_top1 + 1))
            for key in keys:
                if isinstance(top1_camkd[key], list):
                    ax12.plot(epochs_top1, top1_camkd[key][:min_len_top1], 
                             marker='o', label=f'{key} (CAMKD)', linewidth=2)

            ax12.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
            ax12.set_xlabel('Epoch', fontsize=11)
            ax12.set_ylabel('Top-1 Agreement (%)', fontsize=11)
            ax12.set_title('12. Top-1 Agreement: Student vs Teachers ⭐', fontsize=12, fontweight='bold')
            ax12.legend(fontsize=9)
            ax12.grid(True, alpha=0.3)
            ax12.set_ylim([0, 105])

    # ========================================================================
    # 13. Final Top-1 Agreement Bar Chart ⭐
    # ========================================================================
    ax13 = plt.subplot(4, 4, 13)
    if top1_camkd and top1_vanilla and isinstance(top1_camkd, dict):
        keys = sorted(top1_camkd.keys())
        final_camkd_top1 = [top1_camkd[key][-1] if isinstance(top1_camkd[key], list) else 0 for key in keys]
        final_vanilla_top1 = [top1_vanilla.get(key, [0])[-1] if isinstance(top1_vanilla.get(key), list) else 0 for key in keys]

        x = np.arange(len(keys))
        width = 0.35

        bars1 = ax13.bar(x - width/2, final_camkd_top1, width, label='CAMKD', alpha=0.8, color='blue')
        bars2 = ax13.bar(x + width/2, final_vanilla_top1, width, label='Vanilla', alpha=0.8, color='red')

        ax13.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax13.set_ylabel('Agreement (%)', fontsize=11)
        ax13.set_title('13. Final Top-1 Agreement Distribution ⭐', fontsize=12, fontweight='bold')
        ax13.set_xticks(x)
        ax13.set_xticklabels([k.replace('teacher_', 'T') for k in keys], rotation=0)
        ax13.legend(fontsize=10)
        ax13.grid(True, alpha=0.3, axis='y')
        ax13.set_ylim([0, 105])

        for bar in bars1:
            height = bar.get_height()
            ax13.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax13.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # ========================================================================
    # 14. Summary Table
    # ========================================================================
    ax14 = plt.subplot(4, 4, 14)
    ax14.axis('off')

    final_acc_camkd = metrics_camkd['test_acc'][-1]
    final_acc_vanilla = metrics_vanilla['test_acc'][-1]
    acc_improvement = final_acc_camkd - final_acc_vanilla

    centroid_camkd = metrics_camkd.get('centroid_fidelity', [])
    centroid_vanilla = metrics_vanilla.get('centroid_fidelity', [])
    final_centroid_camkd = centroid_camkd[-1] if centroid_camkd else 0
    final_centroid_vanilla = centroid_vanilla[-1] if centroid_vanilla else 0

    top1_avg_camkd = np.mean([v[-1] for v in top1_camkd.values() if isinstance(v, list)]) if top1_camkd else 0
    top1_avg_vanilla = np.mean([v[-1] for v in top1_vanilla.values() if isinstance(v, list)]) if top1_vanilla else 0

    summary_text = f"""
    ╔════════════════════════════╗
    ║   FINAL COMPARISON         ║
    ╚════════════════════════════╝

    Test Accuracy:
      CAMKD:   {final_acc_camkd:.2f}%
      Vanilla: {final_acc_vanilla:.2f}%
      Δ:       {acc_improvement:+.2f}%

    Centroid Fidelity (lower=better):
      CAMKD:   {final_centroid_camkd:.4f}
      Vanilla: {final_centroid_vanilla:.4f}
      Δ:       {final_centroid_camkd - final_centroid_vanilla:+.4f}

    Top-1 Agreement (higher=better) ⭐:
      CAMKD:   {top1_avg_camkd:.1f}%
      Vanilla: {top1_avg_vanilla:.1f}%
      Δ:       {top1_avg_camkd - top1_avg_vanilla:+.1f}%

    WINNER: {("CAMKD 🏆" if acc_improvement > 0 else "Vanilla 🏆" if acc_improvement < 0 else "TIE 🤝")}
    """

    ax14.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_camkd_vs_vanilla.png', 
                dpi=150, bbox_inches='tight')
    print(f"✓ Сохранено: {save_dir}/comparison_camkd_vs_vanilla.png (14 графиков)")
    plt.close()


def print_comparison_statistics(metrics_camkd, metrics_vanilla):
    """Выводит текстовую статистику сравнения"""

    print("\n" + "="*70)
    print("СТАТИСТИКА СРАВНЕНИЯ: CAMKD vs Vanilla KD")
    print("="*70)

    # Test Accuracy
    print(f"\nTest Accuracy (Final):")
    acc_camkd = metrics_camkd['test_acc'][-1]
    acc_vanilla = metrics_vanilla['test_acc'][-1]
    print(f"  CAMKD:   {acc_camkd:.2f}%")
    print(f"  Vanilla: {acc_vanilla:.2f}%")
    print(f"  Δ:       {acc_camkd - acc_vanilla:+.2f}%")

    # Test Loss
    print(f"\nTest Loss (Final):")
    loss_camkd = metrics_camkd['test_loss'][-1]
    loss_vanilla = metrics_vanilla['test_loss'][-1]
    print(f"  CAMKD:   {loss_camkd:.4f}")
    print(f"  Vanilla: {loss_vanilla:.4f}")
    print(f"  Δ:       {loss_camkd - loss_vanilla:+.4f}")

    # Individual Fidelity
    individual_camkd = metrics_camkd.get('individual_fidelity', {})
    individual_vanilla = metrics_vanilla.get('individual_fidelity', {})

    if individual_camkd and individual_vanilla and isinstance(individual_camkd, dict):
        print(f"\nIndividual Fidelity (Final):")
        camkd_values = []
        vanilla_values = []

        for key in sorted(individual_camkd.keys()):
            if isinstance(individual_camkd[key], list) and len(individual_camkd[key]) > 0:
                camkd_val = individual_camkd[key][-1]
                vanilla_val = individual_vanilla.get(key, [0])[-1] if isinstance(individual_vanilla.get(key), list) else 0
                camkd_values.append(camkd_val)
                vanilla_values.append(vanilla_val)
                print(f"  {key}:")
                print(f"    CAMKD:   {camkd_val:.4f}")
                print(f"    Vanilla: {vanilla_val:.4f}")
                print(f"    Δ:       {camkd_val - vanilla_val:+.4f}")

        if camkd_values:
            print(f"  Average:")
            print(f"    CAMKD:   {np.mean(camkd_values):.4f}")
            print(f"    Vanilla: {np.mean(vanilla_values):.4f}")
            print(f"    Δ:       {np.mean(camkd_values) - np.mean(vanilla_values):+.4f}")

    # Centroid Fidelity
    centroid_camkd = metrics_camkd.get('centroid_fidelity', [])
    centroid_vanilla = metrics_vanilla.get('centroid_fidelity', [])

    if centroid_camkd and centroid_vanilla:
        print(f"\nCentroid Fidelity (Final):")
        print(f"  CAMKD:   {centroid_camkd[-1]:.4f}")
        print(f"  Vanilla: {centroid_vanilla[-1]:.4f}")
        print(f"  Δ:       {centroid_camkd[-1] - centroid_vanilla[-1]:+.4f}")

    # Teacher Weights
    teacher_weights_camkd = metrics_camkd.get('teacher_weights', {})
    teacher_weights_vanilla = metrics_vanilla.get('teacher_weights', {})

    if teacher_weights_camkd and isinstance(teacher_weights_camkd, dict):
        print(f"\nTeacher Weights (Final):")
        print(f"  CAMKD (адаптивные):")
        for key in sorted(teacher_weights_camkd.keys()):
            if isinstance(teacher_weights_camkd[key], list) and len(teacher_weights_camkd[key]) > 0:
                print(f"    {key}: {teacher_weights_camkd[key][-1]:.4f}")

        if teacher_weights_vanilla and isinstance(teacher_weights_vanilla, dict):
            print(f"  Vanilla (равные):")
            for key in sorted(teacher_weights_vanilla.keys()):
                if isinstance(teacher_weights_vanilla[key], list) and len(teacher_weights_vanilla[key]) > 0:
                    print(f"    {key}: {teacher_weights_vanilla[key][-1]:.4f}")

    # Top-1 Agreement
    top1_camkd = metrics_camkd.get('top1_agreement', {})
    top1_vanilla = metrics_vanilla.get('top1_agreement', {})

    if top1_camkd and top1_vanilla and isinstance(top1_camkd, dict):
        print(f"\nTop-1 Agreement (Final %) ⭐:")
        camkd_agr = []
        vanilla_agr = []

        for key in sorted(top1_camkd.keys()):
            if isinstance(top1_camkd[key], list) and len(top1_camkd[key]) > 0:
                camkd_val = top1_camkd[key][-1]
                vanilla_val = top1_vanilla.get(key, [0])[-1] if isinstance(top1_vanilla.get(key), list) else 0
                camkd_agr.append(camkd_val)
                vanilla_agr.append(vanilla_val)
                print(f"  {key}:")
                print(f"    CAMKD:   {camkd_val:.1f}%")
                print(f"    Vanilla: {vanilla_val:.1f}%")
                print(f"    Δ:       {camkd_val - vanilla_val:+.1f}%")

        if camkd_agr:
            print(f"  Average:")
            print(f"    CAMKD:   {np.mean(camkd_agr):.1f}%")
            print(f"    Vanilla: {np.mean(vanilla_agr):.1f}%")
            print(f"    Δ:       {np.mean(camkd_agr) - np.mean(vanilla_agr):+.1f}%")

    print("="*70 + "\n")


def compare_experiments(dataset, save_dir='./results'):
    """Главная функция для сравнения экспериментов"""

    dataset_dir = f'{save_dir}/{dataset.lower()}'

    print(f"\n🔍 Загрузка метрик из: {dataset_dir}/")

    # Load metrics
    metrics_camkd = load_metrics(f'{dataset_dir}/metrics_camkd.json')
    metrics_vanilla = load_metrics(f'{dataset_dir}/metrics_vanilla.json')

    if metrics_camkd is None or metrics_vanilla is None:
        print("\n✗ Не удалось загрузить метрики!")
        print("Сначала запустите обучение:")
        print("  python train_student.py --dataset FashionMNIST --methods both")
        return False

    # Infer teacher names
    teacher_names = ['Teacher 1', 'Teacher 2', 'Teacher 3']

    # Print statistics
    print_comparison_statistics(metrics_camkd, metrics_vanilla)

    # Plot comparison
    compare_two_methods(metrics_camkd, metrics_vanilla, teacher_names, dataset_dir)

    print(f"✓ Анализ завершён!")
    print(f"✓ Результаты сохранены в: {dataset_dir}/")

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Анализ результатов Knowledge Distillation')
    parser.add_argument('--dataset', type=str, default='FashionMNIST',
                       choices=['MNIST', 'FashionMNIST', 'CIFAR10'])
    parser.add_argument('--save_dir', type=str, default='./results')

    args = parser.parse_args()

    success = compare_experiments(args.dataset, args.save_dir)

    if not success:
        exit(1)

