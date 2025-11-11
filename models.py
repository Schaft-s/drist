"""
Архитектуры моделей для Knowledge Distillation
5 различных учителей + студент + утилиты для загрузки pretrained моделей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple


# =============================================================================
# УЧИТЕЛЯ (5 разных архитектур)
# =============================================================================

class TeacherCNN1(nn.Module):
    """Глубокая CNN с 4 конволюционными слоями"""
    def __init__(self, num_classes=10):
        super(TeacherCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, is_feat=False):
        x1 = self.pool(F.relu(self.conv1(x)))  # 14x14
        x2 = self.pool(F.relu(self.conv2(x1)))  # 7x7
        x3 = self.pool(F.relu(self.conv3(x2)))  # 3x3
        x4 = self.pool(F.relu(self.conv4(x3)))  # 1x1
        x_flat = x4.view(x4.size(0), -1)
        feat = F.relu(self.fc1(x_flat))
        logits = self.fc2(self.dropout(feat) if self.training else feat)

        if is_feat:
            return [x1, x2, x3, x4, feat], logits
        return logits


class TeacherCNN2(nn.Module):
    """Широкая CNN с большими фильтрами (5x5)"""
    def __init__(self, num_classes=10):
        super(TeacherCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, is_feat=False):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = self.pool(F.relu(self.conv3(x2)))
        x_flat = x3.view(x3.size(0), -1)
        feat = F.relu(self.fc1(x_flat))
        logits = self.fc2(self.dropout(feat) if self.training else feat)

        if is_feat:
            return [x1, x2, x3, feat], logits
        return logits


class TeacherResNet(nn.Module):
    """ResNet-like с skip connections"""
    def __init__(self, num_classes=10):
        super(TeacherResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, is_feat=False):
        # Block 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        identity = x1
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = x1 + identity

        # Block 2
        x2 = F.relu(self.bn3(self.conv3(x1)))
        identity = x2
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2 = x2 + identity

        # Block 3
        x3 = F.relu(self.bn5(self.conv5(x2)))

        x4 = self.pool(x3)
        feat = x4.view(x4.size(0), -1)
        logits = self.fc(feat)

        if is_feat:
            return [x1, x2, x3, feat], logits
        return logits


class TeacherVGG(nn.Module):
    """VGG-like архитектура"""
    def __init__(self, num_classes=10):
        super(TeacherVGG, self).__init__()

        # VGG blocks
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, is_feat=False):
        # Block 1
        x1 = F.relu(self.conv1_1(x))
        x1 = F.relu(self.conv1_2(x1))
        x1 = self.pool(x1)

        # Block 2
        x2 = F.relu(self.conv2_1(x1))
        x2 = F.relu(self.conv2_2(x2))
        x2 = self.pool(x2)

        # Block 3
        x3 = F.relu(self.conv3_1(x2))
        x3 = F.relu(self.conv3_2(x3))
        x3 = self.pool(x3)

        x_flat = x3.view(x3.size(0), -1)
        feat = F.relu(self.fc1(x_flat))
        feat_drop = self.dropout(feat) if self.training else feat
        logits = self.fc2(feat_drop)

        if is_feat:
            return [x1, x2, x3, feat], logits
        return logits


class TeacherDenseNet(nn.Module):
    """DenseNet-like с dense connections"""
    def __init__(self, num_classes=10):
        super(TeacherDenseNet, self).__init__()

        # Initial conv
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(32)

        # Dense block 1
        self.conv1_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        # Transition 1
        self.trans1 = nn.Conv2d(96, 48, 1)
        self.trans1_bn = nn.BatchNorm2d(48)

        # Dense block 2
        self.conv2_1 = nn.Conv2d(48, 48, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(48)
        self.conv2_2 = nn.Conv2d(96, 48, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(48)

        # Transition 2
        self.trans2 = nn.Conv2d(144, 72, 1)
        self.trans2_bn = nn.BatchNorm2d(72)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(72, num_classes)

    def forward(self, x, is_feat=False):
        # Initial
        x0 = F.relu(self.bn0(self.conv0(x)))

        # Dense block 1
        x1_1 = F.relu(self.bn1_1(self.conv1_1(x0)))
        x1_cat1 = torch.cat([x0, x1_1], dim=1)

        x1_2 = F.relu(self.bn1_2(self.conv1_2(x1_cat1)))
        x1_cat2 = torch.cat([x1_cat1, x1_2], dim=1)

        # Transition 1
        x1_trans = self.pool(F.relu(self.trans1_bn(self.trans1(x1_cat2))))

        # Dense block 2
        x2_1 = F.relu(self.bn2_1(self.conv2_1(x1_trans)))
        x2_cat1 = torch.cat([x1_trans, x2_1], dim=1)

        x2_2 = F.relu(self.bn2_2(self.conv2_2(x2_cat1)))
        x2_cat2 = torch.cat([x2_cat1, x2_2], dim=1)

        # Transition 2
        x2_trans = self.pool(F.relu(self.trans2_bn(self.trans2(x2_cat2))))

        # Global pooling
        feat = self.global_pool(x2_trans)
        feat = feat.view(feat.size(0), -1)

        logits = self.fc(feat)

        if is_feat:
            return [x0, x1_cat2, x2_cat2, feat], logits
        return logits


# =============================================================================
# СТУДЕНТ
# =============================================================================

class StudentNet(nn.Module):
    """Легкая модель студента"""
    def __init__(self, num_classes=10):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, is_feat=False):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x_flat = x2.view(-1, 32 * 7 * 7)
        feat = F.relu(self.fc1(x_flat))
        logits = self.fc2(self.dropout(feat) if self.training else feat)

        if is_feat:
            return [x1, x2, feat], logits
        return logits


# =============================================================================
# CONVREG для feature matching
# =============================================================================

class ConvReg(nn.Module):
    """Convolutional regression для согласования размерностей features"""
    def __init__(self, s_shape, t_shape):
        super(ConvReg, self).__init__()
        self.use_conv = len(s_shape) > 2

        if self.use_conv:
            s_C, s_H, s_W = s_shape[1], s_shape[2], s_shape[3]
            t_C, t_H, t_W = t_shape[1], t_shape[2], t_shape[3]

            if s_C != t_C:
                self.conv = nn.Conv2d(s_C, t_C, kernel_size=1, stride=1, padding=0)
            else:
                self.conv = None

            if s_H != t_H or s_W != t_W:
                self.pool = nn.AdaptiveAvgPool2d((t_H, t_W))
            else:
                self.pool = None
        else:
            s_C = s_shape[1]
            t_C = t_shape[1]
            if s_C != t_C:
                self.fc = nn.Linear(s_C, t_C)
            else:
                self.fc = None

    def forward(self, x):
        if self.use_conv:
            if self.conv is not None:
                x = self.conv(x)
            if self.pool is not None:
                x = self.pool(x)
        else:
            if hasattr(self, 'fc') and self.fc is not None:
                x = self.fc(x)
        return x


# =============================================================================
# PRETRAINED MODEL URLS (для загрузки)
# =============================================================================

PRETRAINED_URLS = {
    'mnist': {
        'teacher_cnn1': 'https://github.com/example/pretrained/mnist_cnn1.pth',
        'teacher_cnn2': 'https://github.com/example/pretrained/mnist_cnn2.pth',
        'teacher_resnet': 'https://github.com/example/pretrained/mnist_resnet.pth',
    },
    'fashion_mnist': {
        'teacher_cnn1': 'https://github.com/example/pretrained/fashion_cnn1.pth',
        'teacher_cnn2': 'https://github.com/example/pretrained/fashion_cnn2.pth',
        'teacher_resnet': 'https://github.com/example/pretrained/fashion_resnet.pth',
    },
    'cifar10': {
        'teacher_cnn1': 'https://github.com/example/pretrained/cifar10_cnn1.pth',
        'teacher_cnn2': 'https://github.com/example/pretrained/cifar10_cnn2.pth',
        'teacher_resnet': 'https://github.com/example/pretrained/cifar10_resnet.pth',
    }
}


def load_pretrained_teacher(model_name, dataset, num_classes=10, device='cpu'):
    """
    Загружает предобученную модель учителя

    Args:
        model_name: название модели ('teacher_cnn1', 'teacher_cnn2', etc.)
        dataset: датасет ('mnist', 'fashion_mnist', 'cifar10')
        num_classes: количество классов
        device: устройство для загрузки

    Returns:
        model: загруженная модель
        accuracy: точность модели (если есть в checkpoint)
    """
    import os
    import torch

    # Создаём модель
    model_dict = {
        'teacher_cnn1': TeacherCNN1,
        'teacher_cnn2': TeacherCNN2,
        'teacher_resnet': TeacherResNet,
        'teacher_vgg': TeacherVGG,
        'teacher_densenet': TeacherDenseNet,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_dict[model_name](num_classes=num_classes)

    # Пробуем загрузить локально
    local_path = f'./pretrained/{dataset}/{model_name}.pth'

    if os.path.exists(local_path):
        print(f"Loading from local: {local_path}")
        checkpoint = torch.load(local_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('test_acc', None)
        print(f"✓ Loaded {model_name} from local (Acc: {accuracy:.2f}%)")
        model = model.to(device)
        model.eval()
        return model, accuracy

    # Если нет локально, пробуем скачать
    if dataset in PRETRAINED_URLS and model_name in PRETRAINED_URLS[dataset]:
        url = PRETRAINED_URLS[dataset][model_name]
        print(f"Attempting to download from: {url}")
        # TODO: implement download
        print(f"⚠️ Download not implemented yet. Please train teachers first.")
        return None, None

    print(f"⚠️ No pretrained model found for {model_name} on {dataset}")
    print(f"   Expected path: {local_path}")
    print(f"   Please train teachers first: python train_teachers.py")
    return None, None


# =============================================================================
# MODEL REGISTRY
# =============================================================================

model_dict = {
    'teacher_cnn1': TeacherCNN1,
    'teacher_cnn2': TeacherCNN2,
    'teacher_resnet': TeacherResNet,
    'teacher_vgg': TeacherVGG,
    'teacher_densenet': TeacherDenseNet,
    'student': StudentNet,
}


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info():
    """Выводит информацию о всех доступных моделях"""
    print("\n" + "="*70)
    print("ДОСТУПНЫЕ МОДЕЛИ")
    print("="*70)

    for name, model_class in model_dict.items():
        model = model_class(num_classes=10)
        params = count_parameters(model)
        print(f"{name:20s}: {params:>10,} параметров")

    print("="*70 + "\n")


if __name__ == '__main__':
    get_model_info()
