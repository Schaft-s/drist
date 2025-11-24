"""
Архитектуры моделей студента для Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentNet(nn.Module):
    """Легкая модель студента"""
    def __init__(self, num_classes=10):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(128)
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

class MicroStudentNet(nn.Module):
    """
    Очень маленький студент (~50k параметров, порядок величины).
    Для FashionMNIST: 1×28×28.
    """
    def __init__(self, num_classes=10):
        super(MicroStudentNet, self).__init__()
        # меньше каналов, меньше скрытое представление
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)   # 1→4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)   # 4→8
        self.pool = nn.MaxPool2d(2, 2)  # 28→14→7
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.LazyLinear(32)   # маленький bottleneck
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, is_feat=False):
        x = F.relu(self.conv1(x))      # [B, 4, 28, 28]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 8, 14, 14] -> pool -> [B, 8, 7, 7]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        feat = F.relu(self.fc1(x))     # [B, 32]
        x = self.dropout2(feat)
        logits = self.fc2(x)

        if is_feat:
            # Чтобы быть совместимым с учителями, возвращаем список фичей
            return [feat], logits
        return logits

class TinyStudentNet(nn.Module):
    """
    Урезанная версия студента (меньше каналов).
    Цель: создать 'bottle-neck', чтобы студент не мог выучить датасет идеально сам.
    Параметров: ~12k (в 17 раз меньше, чем у обычного студента)
    """
    def __init__(self, num_classes=10):
        super(TinyStudentNet, self).__init__()
        # Было 32 и 64 канала -> Стало 8 и 16
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # LazyLinear автоматически определит размер входа
        self.fc1 = nn.LazyLinear(64) 
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, is_feat=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        f = self.fc1(x)
        feat = F.relu(f)
        
        x = self.dropout2(feat)
        x = self.fc2(x)
        
        if is_feat:
            return [f], x # Возвращаем список фичей для совместимости
        return x

student_model_dict = {
    'student': StudentNet, 'tiny_student': TinyStudentNet, 'micro_student': MicroStudentNet
}



def count_parameters(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Подсчитывает параметры с инициализацией LazyLinear"""
    import torch.nn as nn
    
    has_lazy = any(isinstance(m, nn.modules.lazy.LazyModuleMixin) for m in model.modules())
    
    if has_lazy:
        try:
            dummy_input = torch.randn(1, 1, 28, 28).to(device)
            with torch.no_grad():
                if hasattr(model, 'is_feat'):
                    model(dummy_input, is_feat=False)
                else:
                    model(dummy_input)
        except Exception:
            try:
                dummy_input = torch.randn(1, 3, 32, 32).to(device)
                with torch.no_grad():
                    if hasattr(model, 'is_feat'):
                        model(dummy_input, is_feat=False)
                    else:
                        model(dummy_input)
            except Exception:
                pass
    
    total_params = 0
    for param in model.parameters():
        try:
            if param.requires_grad:
                total_params += param.numel()
        except ValueError:
            pass
    
    return total_params


if __name__ == '__main__':
    model = StudentNet(num_classes=10)
    params = count_parameters(model)
    print(f"StudentNet: {params:,} параметров")

