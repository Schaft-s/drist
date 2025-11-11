# 🚀 Quick Start для Google Colab

## Шаг 1: Клонирование репозитория

```python
# В ячейке Colab
!git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ> drist
%cd drist
```

Или загрузите файлы вручную через интерфейс Colab.

## Шаг 2: Установка зависимостей

```python
!pip install torch torchvision matplotlib numpy pyyaml -q
```

## Шаг 3: Обучение учителей

```python
# Обучение 3 учителей на FashionMNIST
!python train_teachers/train_teachers.py --config train_teachers/configs/fashionmnist_3teachers.yaml
```

Доступные конфиги:
- `train_teachers/configs/mnist_3teachers.yaml` - MNIST, 3 учителя
- `train_teachers/configs/fashionmnist_3teachers.yaml` - FashionMNIST, 3 учителя
- `train_teachers/configs/fashionmnist_5teachers.yaml` - FashionMNIST, 5 учителей
- `train_teachers/configs/cifar10_3teachers.yaml` - CIFAR10, 3 учителя

## Шаг 4: Обучение студента

```python
# Обучение студента обоими методами (CAMKD + Vanilla)
!python train_student/train_student.py --config train_student/configs/fashionmnist_both.yaml
```

Доступные конфиги:
- `train_student/configs/fashionmnist_both.yaml` - оба метода на FashionMNIST
- `train_student/configs/fashionmnist_camkd.yaml` - только CAMKD
- `train_student/configs/fashionmnist_vanilla.yaml` - только Vanilla
- `train_student/configs/mnist_both.yaml` - оба метода на MNIST

## Шаг 5: Анализ результатов

```python
# Сравнение методов
!python -m utils.analysis --dataset FashionMNIST
```

## Полный цикл (одной командой)

```python
# 1. Обучение учителей
!python train_teachers/train_teachers.py --config train_teachers/configs/fashionmnist_3teachers.yaml

# 2. Обучение студента
!python train_student/train_student.py --config train_student/configs/fashionmnist_both.yaml

# 3. Анализ
!python -m utils.analysis --dataset FashionMNIST
```

## Быстрый тест (меньше эпох)

Создайте свой конфиг или измените существующий:

```yaml
# quick_test.yaml
dataset: MNIST
epochs: 3  # Быстрое обучение
batch_size: 128
lr: 0.001
save_dir: ./pretrained
teachers:
  - teacher_cnn1
  - teacher_cnn2
  - teacher_resnet
```

```python
# Обучение учителей (3 эпохи)
!python train_teachers/train_teachers.py --config quick_test.yaml

# Обучение студента (5 эпох)
# Создайте аналогичный конфиг для студента с epochs: 5
!python train_student/train_student.py --config student_quick_test.yaml
```

## Структура проекта

```
drist/
├── train_teachers/
│   ├── train_teachers.py      # Скрипт обучения
│   ├── teacher_models.py       # Модели учителей
│   └── configs/                # Конфигурации
│       ├── fashionmnist_3teachers.yaml
│       └── ...
├── train_student/
│   ├── train_student.py        # Скрипт обучения
│   ├── student_models.py       # Модели студентов
│   ├── convreg.py              # ConvReg для features
│   └── configs/                # Конфигурации
│       ├── fashionmnist_both.yaml
│       └── ...
├── utils/
│   ├── utils.py                # Утилиты (метрики, визуализация)
│   ├── distillation_losses.py  # Loss функции
│   └── analysis.py             # Анализ результатов
└── scripts/                    # Скрипты для Colab
    ├── colab_setup.sh
    ├── colab_train_teachers.sh
    └── ...
```

## Результаты

После выполнения всех шагов вы получите:

```
pretrained/
  └── fashionmnist/
      ├── teacher_cnn1.pth
      ├── teacher_cnn2.pth
      └── teacher_resnet.pth

results/
  └── fashionmnist/
      ├── student_camkd.pth
      ├── student_vanilla.pth
      ├── metrics_camkd.json
      ├── metrics_vanilla.json
      ├── plots_camkd.png
      ├── plots_vanilla.png
      └── comparison_camkd_vs_vanilla.png
```

## Кастомизация конфигов

Вы можете создать свои конфиги, скопировав существующие и изменив параметры:

```yaml
dataset: FashionMNIST
epochs: 20
batch_size: 128
lr: 0.001
temperature: 4.0
alpha: 0.7
beta: 100.0
# ... и т.д.
```

## Troubleshooting

### Ошибка импорта
```python
# Убедитесь, что вы в корневой директории проекта
import sys
sys.path.insert(0, '.')
```

### CUDA out of memory
Уменьшите `batch_size` в конфиге:
```yaml
batch_size: 64  # вместо 128
```

### Файлы не найдены
Проверьте пути в конфигах и убедитесь, что учителя обучены перед обучением студента.

---

**Готово! 🎉**

