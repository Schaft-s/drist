# 🎓 Knowledge Distillation: CAMKD vs Vanilla KD

Полный набор для экспериментов с Multi-Teacher Knowledge Distillation.

## 📁 Структура проекта

```
.
├── train_teachers/              # Модуль обучения учителей
│   ├── train_teachers.py        # Скрипт обучения учителей
│   ├── teacher_models.py        # 5 архитектур учителей
│   ├── __init__.py
│   └── configs/                 # Конфигурации для обучения
│       ├── mnist_3teachers.yaml
│       ├── fashionmnist_3teachers.yaml
│       ├── fashionmnist_5teachers.yaml
│       └── cifar10_3teachers.yaml
│
├── train_student/               # Модуль обучения студентов
│   ├── train_student.py         # Скрипт обучения студентов
│   ├── student_models.py        # Модель студента
│   ├── convreg.py               # ConvReg для feature matching
│   ├── __init__.py
│   └── configs/                 # Конфигурации для обучения
│       ├── fashionmnist_both.yaml
│       ├── fashionmnist_camkd.yaml
│       ├── fashionmnist_vanilla.yaml
│       └── mnist_both.yaml
│
├── utils/                       # Утилиты
│   ├── utils.py                 # Метрики fidelity + визуализация
│   ├── distillation_losses.py   # CAMKD + DistillKL losses
│   ├── analysis.py              # Анализ и сравнение результатов
│   └── __init__.py
│
├── scripts/                     # Скрипты для Google Colab
│   ├── colab_setup.sh
│   ├── colab_train_teachers.sh
│   ├── colab_train_student.sh
│   └── colab_analyze.sh
│
├── COLAB_QUICKSTART.md          # Инструкция для Google Colab
└── README.md                    # Эта инструкция
```

## 🏗️ Архитектуры

### Учителя (5 разных):

1. **TeacherCNN1** (276K params)
   - Глубокая CNN с 4 conv слоями
   - MaxPool, Dropout 0.5

2. **TeacherCNN2** (1.7M params)  
   - Широкая CNN с фильтрами 5x5
   - Большая capacity

3. **TeacherResNet** (141K params)
   - ResNet-like с skip connections
   - BatchNorm

4. **TeacherVGG** (359K params)
   - VGG-style с двойными conv блоками
   - Dropout 0.5

5. **TeacherDenseNet** (218K params)
   - DenseNet-like с dense connections
   - Concatenation features

### Студент:

**StudentNet** (207K params)
- Легкая 2-слойная CNN
- Dropout 0.25

## 🚀 Быстрый старт

### Локально

#### 1. Обучение учителей

```bash
python train_teachers/train_teachers.py --config train_teachers/configs/fashionmnist_3teachers.yaml
```

#### 2. Обучение студента

```bash
python train_student/train_student.py --config train_student/configs/fashionmnist_both.yaml
```

#### 3. Анализ результатов

```bash
python -m utils.analysis --dataset FashionMNIST
```

### В Google Colab

См. подробную инструкцию в [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)

```python
# 1. Клонирование
!git clone <URL> drist
%cd drist

# 2. Установка зависимостей
!pip install torch torchvision matplotlib numpy pyyaml -q

# 3. Обучение учителей
!python train_teachers/train_teachers.py --config train_teachers/configs/fashionmnist_3teachers.yaml

# 4. Обучение студента
!python train_student/train_student.py --config train_student/configs/fashionmnist_both.yaml

# 5. Анализ
!python -m utils.analysis --dataset FashionMNIST
```

## 📊 Методы дистилляции

### 1. CAMKD (Cross-teacher Attentive Multi-teacher KD)

**Особенности:**
- Адаптивные веса учителей: `w_i = (1 - softmax(loss_t)) / (M-1)`
- Feature distillation на промежуточных слоях
- MSE между student и teacher features

**Loss:**
```
Loss = CE(student, labels) 
     + α·KL(student, avg_teacher)
     + β·CAMKD_feature
```

### 2. Vanilla KD

**Особенности:**
- Равные веса всех учителей (1/M)
- Только logit distillation (без features)
- Простое усреднение выходов учителей

**Loss:**
```
Loss = CE(student, labels) 
     + α·KL(student, avg_teacher)
```

## 📈 Отслеживаемые метрики

### 1. Individual Fidelity
```
KL(Teacher_i || Student) для каждого учителя
```

### 2. Centroid Fidelity
```
KL(Centroid || Student)
где Centroid = (T1 + T2 + T3) / M
```

### 3. Teacher Diversity
```
KL(Teacher_i || Centroid) для каждого учителя
```

### 4. Pairwise Diversity
```
KL(Teacher_i || Teacher_j) для всех пар
```

### 5. Top-1 Agreement
```
Процент совпадения предсказаний студента и учителей
```

## 🎨 Визуализация

### Автоматически создаваемые графики:

#### После train_student.py:

1. `plots_camkd.png` - 9 базовых графиков (CAMKD)
2. `plots_vanilla.png` - 9 базовых графиков (Vanilla)

#### После analysis.py:

3. `comparison_camkd_vs_vanilla.png` - 14 графиков сравнения:
   - Test Accuracy / Loss
   - Individual Fidelity (по каждому учителю)
   - Centroid Fidelity
   - Average Individual Fidelity
   - Teacher Weights
   - Final Metrics Bar Chart
   - Fidelity Distribution
   - Top-1 Agreement
   - Summary Statistics

## 🔧 Конфигурационные файлы

Проект использует YAML конфигурации для управления параметрами экспериментов.

### Пример конфига для учителей:

```yaml
dataset: FashionMNIST
epochs: 20
batch_size: 128
lr: 0.001
save_dir: ./pretrained
teachers:
  - teacher_cnn1
  - teacher_cnn2
  - teacher_resnet
```

### Пример конфига для студента:

```yaml
dataset: FashionMNIST
epochs: 15
batch_size: 128
lr: 0.001
temperature: 4.0
alpha: 0.7
beta: 100.0
hint_layer: -2
teacher_dir: ./pretrained
save_dir: ./results
fidelity_freq: 1
methods: both
teacher_names:
  - teacher_cnn1
  - teacher_cnn2
  - teacher_resnet
use_pretrained: false
```

## 📂 Выходные файлы

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

## 💡 Примеры использования

### Использовать других учителей

Создайте новый конфиг или измените существующий:

```yaml
teachers:
  - teacher_cnn1
  - teacher_cnn2
  - teacher_resnet
  - teacher_vgg
  - teacher_densenet
```

### Только один метод

Используйте конфиг с `methods: camkd` или `methods: vanilla`

### Изменить гиперпараметры

Отредактируйте конфиг:

```yaml
alpha: 0.9
beta: 200
temperature: 6.0
```

## 🎯 Ожидаемые результаты на FashionMNIST

| Модель | Параметры | Точность |
|--------|-----------|----------|
| Teacher CNN1 | 276K | ~92.5% |
| Teacher CNN2 | 1.7M | ~93.3% |
| Teacher ResNet | 141K | ~92.0% |
| **Student (CAMKD)** | **207K** | **~91.5-91.8%** |
| **Student (Vanilla)** | **207K** | **~91.2-91.5%** |

**Ожидаемая разница:** CAMKD на 0.2-0.5% лучше Vanilla

## 📚 Референсы

1. **CAMKD**: "Cross-teacher Attentive Multi-teacher Knowledge Distillation"
2. **Hinton et al. (2015)**: "Distilling the Knowledge in a Neural Network"
3. **Teacher Diversity**: Метрики из постановки задачи

## 🔍 Детальная документация

- [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md) - Подробная инструкция для Google Colab

---

**Готово к экспериментам! 🚀**
