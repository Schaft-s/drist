#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Colab Launcher для экспериментов с Knowledge Distillation
Запускает все эксперименты последовательно
Использует конфигурационные файлы
"""

import subprocess
import os
import sys


def run_command(cmd, description):
    """Запускает команду и выводит результат"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    print(f"Команда: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Ошибка при выполнении: {description}")
        return False

    print(f"\n✅ Успешно завершено: {description}")
    return True


def experiment_1_train_and_distill(dataset='FashionMNIST'):
    """
    Эксперимент 1: Полный цикл
    1. Обучение учителей с нуля
    2. Дистилляция двумя методами (CAMKD + Vanilla)
    3. Анализ и сравнение
    """
    print("\n" + "="*70)
    print("📊 ЭКСПЕРИМЕНТ 1: Полный цикл (Train + Distill + Compare)")
    print("="*70)

    dataset_lower = dataset.lower()
    
    # Step 1: Train teachers
    teacher_config = f"train_teachers/configs/{dataset_lower}_3teachers.yaml"
    if not run_command(
        f"python train_teachers/train_teachers.py --config {teacher_config}",
        "Шаг 1/3: Обучение учителей"
    ):
        return False

    # Step 2: Train students with both methods
    student_config = f"train_student/configs/{dataset_lower}_both.yaml"
    if not run_command(
        f"python train_student/train_student.py --config {student_config}",
        "Шаг 2/3: Обучение студентов (CAMKD + Vanilla)"
    ):
        return False

    # Step 3: Analyze results
    if not run_command(
        f"python -m utils.analysis --dataset {dataset}",
        "Шаг 3/3: Анализ результатов"
    ):
        return False

    print("\n" + "="*70)
    print("✅ ЭКСПЕРИМЕНТ 1 ЗАВЕРШЁН!")
    print("="*70)
    return True


def experiment_2_pretrained_distill(dataset='FashionMNIST'):
    """
    Эксперимент 2: Использование предобученных учителей
    1. Загрузка pretrained учителей
    2. Дистилляция двумя методами
    3. Анализ и сравнение
    """
    print("\n" + "="*70)
    print("📊 ЭКСПЕРИМЕНТ 2: Pretrained учителя (Distill + Compare)")
    print("="*70)

    # Check if pretrained available
    pretrained_dir = f"./pretrained/{dataset.lower()}"
    if not os.path.exists(pretrained_dir):
        print(f"\n⚠️ Предобученные модели не найдены в: {pretrained_dir}")
        print("Запустите сначала эксперимент 1 или загрузите pretrained модели")
        return False

    dataset_lower = dataset.lower()
    
    # Step 1: Train students with pretrained teachers
    student_config = f"train_student/configs/{dataset_lower}_both.yaml"
    if not run_command(
        f"python train_student/train_student.py --config {student_config}",
        "Шаг 1/2: Обучение студентов с pretrained учителями"
    ):
        return False

    # Step 2: Analyze
    if not run_command(
        f"python -m utils.analysis --dataset {dataset}",
        "Шаг 2/2: Анализ результатов"
    ):
        return False

    print("\n" + "="*70)
    print("✅ ЭКСПЕРИМЕНТ 2 ЗАВЕРШЁН!")
    print("="*70)
    return True


def experiment_quick_test():
    """
    Быстрый тест: 3 эпохи учителей + 5 эпох студентов
    Для проверки что всё работает
    """
    print("\n" + "="*70)
    print("⚡ БЫСТРЫЙ ТЕСТ (Quick Test)")
    print("="*70)

    dataset = 'MNIST'

    # Train teachers (quick) - используем существующий конфиг, но можно создать quick версию
    teacher_config = "train_teachers/configs/mnist_3teachers.yaml"
    if not run_command(
        f"python train_teachers/train_teachers.py --config {teacher_config}",
        "Шаг 1/3: Обучение учителей (используйте epochs: 3 в конфиге для быстрого теста)"
    ):
        return False

    # Train students
    student_config = "train_student/configs/mnist_both.yaml"
    if not run_command(
        f"python train_student/train_student.py --config {student_config}",
        "Шаг 2/3: Обучение студентов (используйте epochs: 5 в конфиге для быстрого теста)"
    ):
        return False

    # Analyze
    if not run_command(
        f"python -m utils.analysis --dataset {dataset}",
        "Шаг 3/3: Анализ"
    ):
        return False

    print("\n" + "="*70)
    print("✅ БЫСТРЫЙ ТЕСТ ЗАВЕРШЁН!")
    print("="*70)
    return True


def run_all_experiments():
    """Запускает все эксперименты последовательно"""

    print("\n" + "="*80)
    print("🎯 ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ")
    print("="*80)

    results = {}

    # Quick test
    print("\n[1/2] Запуск быстрого теста...")
    results['quick_test'] = experiment_quick_test()

    # Experiment 1: Full cycle on FashionMNIST
    print("\n[2/2] Запуск полного эксперимента на FashionMNIST...")
    results['experiment_1'] = experiment_1_train_and_distill('FashionMNIST')

    # Summary
    print("\n" + "="*80)
    print("📊 ИТОГОВАЯ СВОДКА")
    print("="*80)
    for name, success in results.items():
        status = "✅ Успешно" if success else "❌ Ошибка"
        print(f"  {name:20s}: {status}")
    print("="*80 + "\n")


def main():
    """Главная функция с меню"""

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\n" + "="*70)
        print("🎯 LAUNCHER: Knowledge Distillation Experiments")
        print("="*70)
        print("\nВыберите режим:")
        print("  1 - Быстрый тест (MNIST)")
        print("  2 - Эксперимент 1: Train + Distill (FashionMNIST)")
        print("  3 - Эксперимент 2: Pretrained + Distill (FashionMNIST)")
        print("  4 - Запустить всё последовательно")
        print("  q - Выход")
        print("="*70)

        mode = input("\nВаш выбор: ").strip()

    if mode == '1' or mode == 'quick':
        experiment_quick_test()

    elif mode == '2' or mode == 'exp1':
        dataset = input("Датасет (MNIST/FashionMNIST/CIFAR10) [FashionMNIST]: ").strip()
        if not dataset:
            dataset = 'FashionMNIST'
        experiment_1_train_and_distill(dataset)

    elif mode == '3' or mode == 'exp2':
        dataset = input("Датасет (MNIST/FashionMNIST/CIFAR10) [FashionMNIST]: ").strip()
        if not dataset:
            dataset = 'FashionMNIST'
        experiment_2_pretrained_distill(dataset)

    elif mode == '4' or mode == 'all':
        run_all_experiments()

    elif mode == 'q' or mode == 'quit':
        print("Выход...")
        return

    else:
        print(f"\n❌ Неизвестный режим: {mode}")
        print("Используйте: 1, 2, 3, 4, или q")


if __name__ == '__main__':
    main()
