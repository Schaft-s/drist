#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОЛНЫЙ СКРИПТ ЗАПУСКА ВСЕГО ЭКСПЕРИМЕНТА
1. Обучение 5 учителей
2. Параллельное обучение 4 студентов на одинаковых батчах
3. Анализ результатов
"""

import subprocess
import os
import sys

def run_command(cmd, description):
    """Запускает команду"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Команда: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nОшибка при выполнении: {description}")
        return False
    
    print(f"\n✅ Успешно завершено: {description}")
    return True

def main():
    print("\n" + "="*80)
    print("ПОЛНЫЙ ЭКСПЕРИМЕНТ: 5 УЧИТЕЛЕЙ + 4 СТУДЕНТА (ПАРАЛЛЕЛЬНО)")
    print("="*80)
    
    dataset = 'FashionMNIST'
    
    # ЭТАП 1: Обучаем 5 учителей
    # (можно пропустить, если уже обучили)
    # if not run_command(
    #     f"python train_teachers/train_teachers.py --config train_teachers/configs/fashionmnist_5teachers.yaml",
    #     f"Этап 1/3: Обучение 5 учителей на {dataset} (25 эпох)"
    # ):
    #     print("Ошибка при обучении учителей!")
    #     return False
    
    # ЭТАП 2: Обучаем 4 студентов ПАРАЛЛЕЛЬНО
    if not run_command(
        f"python train_student/train_student_4methods_parallel.py --config train_student/configs/fashionmnist_4methods.yaml",
        f"Этап 2/3: Параллельное обучение 4 студентов на {dataset} (20 эпох)"
    ):
        print("Ошибка при обучении студентов!")
        return False
    
    # ЭТАП 3: Анализ результатов
    if not run_command(
        f"python utils/analysis_4methods.py --dataset {dataset} --save_dir ./results",
        f"Этап 3/3: Анализ результатов (12 графиков)"
    ):
        print("Ошибка при анализе!")
        return False
    
    print("\n" + "="*80)
    print("ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    print("="*80)
    print(f"\nРезультаты сохранены в: ./results/{dataset.lower()}/")
    print(f"   - comparison_4methods.png (12 графиков)")
    print(f"   - metrics_centroid.json")
    print(f"   - metrics_best.json")
    print(f"   - metrics_median.json")
    print(f"   - metrics_camkd.json")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
