"""
Модуль для обучения студентов
"""

from .student_models import student_model_dict, count_parameters
from .convreg import ConvReg

__all__ = ['student_model_dict', 'count_parameters', 'ConvReg']

