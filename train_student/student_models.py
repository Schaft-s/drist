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


# Model registry
student_model_dict = {
    'student': StudentNet,
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

