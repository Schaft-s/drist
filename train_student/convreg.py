"""
ConvReg для feature matching между student и teacher моделями
"""

import torch
import torch.nn as nn


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

