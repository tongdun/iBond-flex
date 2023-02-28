#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond
#
#  File name: loss.py
#
#  Create date: 2020/12/15
#
from abc import ABCMeta, abstractmethod

import numpy as np
from enum import Enum, unique
import torch

from caffeine.utils import activations
from caffeine.utils.common_tools import *
from caffeine.utils.exceptions import ShapeMismatchError, LossTypeError


@unique
class Loss(Enum):
    '''
    This is Enum class for regulate string when we call losses
    '''
    BCELoss = 'BCELoss'
    MSELoss = 'MSELoss'
    HingeLoss = 'HingeLoss'


class BaseLoss(metaclass=ABCMeta):
    def __init__(self, labels: np.ndarray = np.array([]), reduction: str = 'mean'):
        if reduction in ['mean', 'sum', 'none']:
            self.reduction = reduction
        else:
            raise ValueError(f'Invalid value {reduction} for processing loss outputs')

    @abstractmethod
    def loss(self, predict: np.ndarray, label: np.ndarray) -> Union:
        """
        Compute mean loss.

        Returns:
            Mean loss.
        """
        pass

    @abstractmethod
    def gradient(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Compute gradients.

        Returns:
            corresponding gradients
        """
        pass

    @abstractmethod
    def hessian(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        """
        Compute hessians.

        Returns:
            corresponding hessians
        """
        pass


class ClassificationLoss(BaseLoss):
    """
    Abstract class for all loss
    """
    #todo 加上正则
    def __init__(self, labels: np.ndarray = np.array([]), reduction: str = 'mean'):
        '''

        Args:
            labels: [Option] shape (n,) or (n,1) or (n,2) or (n, m), np.ndarray. Labels for computing class weight.
            reduction: [Option] str, e.g. 'mean', 'sum', 'none', else raise ValueError Exception.
                'mean': return mean of losses.
                'sum': return sum of losses.
                'none: return losses with same length with inputs.

        '''
        super().__init__(labels, reduction)
        if labels.size != 0:
            self.class_weights = get_class_weight(labels)
        else:
            self.class_weights = dict()

    def load_class_weights(self, class_weights):
        self.class_weights = class_weights

    def _convert_label_to_class_weights(self, labels: np.ndarray) -> np.ndarray:
        '''
        Convert input label to class weight.

        Args:
            labels: shape (n,) or (n,1) or (n,2) numpy array. Labels for calc loss/grad/hess

        Returns:
            class_weight: shape (n,) np.ndarray. Flattened class weights for correspoinding labels
        '''
        labels = mcl_repr(labels, squeeze=True)
        return np.array([self.class_weights[i] for i in labels])

    def loss(self, predict: np.ndarray, label: np.ndarray) -> Union:
        raise NotImplementedError()

    def gradient(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def hessian(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class BCELoss(ClassificationLoss):
    def __init__(self, labels: np.ndarray = np.array([]), reduction: str = 'mean'):
        '''

        Args:
            labels: [Option] shape (n,) or (n,1) or (n,2), np.ndarray. Labels for computing class weight.
            reduction: [Option] str, e.g. 'mean', 'sum', 'none', else raise ValueError Exception.
                'mean': return mean of losses.
                'sum': return sum of losses.
                'none: return losses with same length with inputs.

        '''
        super(BCELoss, self).__init__(labels, reduction)

    def loss(self, predict: np.ndarray, label: np.ndarray) -> Union:
        """
        Compute BCE Loss

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.

        Returns:
            loss: Union: float or (n,) shape numpy array. BCE Loss
        """
        predict = bcl_repr(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        if self.class_weights:
            weights = self._convert_label_to_class_weights(label)
            losses = - weights * (label * np.log(predict) + (1 - label) * np.log(1 - predict))
        else:
            losses = - label * np.log(predict + 1e-15) - (1. - label) * np.log(1. + 1e-15 - predict)
        if self.reduction == 'mean':
            return float(np.mean(losses.flatten()))
        elif self.reduction == 'sum':
            return float(np.sum(losses.flatten()))
        else:
            return losses.flatten()

    def gradient(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:  # todo 这里是有问题的。
        '''
        Compute gradients.

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.

        Returns:
            gradients: shape (n,) np.ndarray. Gradients with same length like input data.
        '''
        predict = bcl_repr(predict)
        predict = activations.sigmoid(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        if self.class_weights:
            weights = self._convert_label_to_class_weights(label)
            return weights * (predict - label)
            # 还是有问题，要改成能够改成特定值的时候，和没有weight应该是一样的
        else:
            return predict - label

    def hessian(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Compute hessians.

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.abel:

        Returns:
            hessians: shape (n,) np.ndarray. Hessians with same length like input data.
        '''
        predict = bcl_repr(predict)
        predict = activations.sigmoid(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        if self.class_weights:
            weights = self._convert_label_to_class_weights(label)
            return weights * (predict * (1 - predict))
            # 还是有问题，要改成能够改成特定值的时候，和没有weight应该是一样的
        else:
            return predict * (1 - predict)

    def predict(self, value):
        value = value.astype(float)
        index1 = np.where(value <= 0)
        index2 = np.where(value > 0)
        temp1 = np.exp(value[index1])
        value[index1] = temp1 / (1. + temp1)
        value[index2] = 1. / (1. + np.exp(-value[index2]))
        return value


class CELoss(ClassificationLoss):
    def __init__(self, labels: np.ndarray = np.array([]), reduction: str = 'mean', classes: List = []):
        '''

        Args:
            labels: [Option] shape (n,) or (n,1) or (n,2), np.ndarray. Labels for computing class weight.
            reduction: [Option] str, e.g. 'mean', 'sum', 'none', else raise ValueError Exception.
                'mean': return mean of losses.
                'sum': return sum of losses.
                'none: return losses with same length with inputs.

        '''
        super(CELoss, self).__init__(labels, reduction)
        self.classes = classes

    def loss(self, predict: np.ndarray, label: np.ndarray) -> Union:
        """
        Compute CE Loss

        Args:
            predict: shape (n,) or (n,1) or (n,m) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,m) numpy array

        Returns:
            loss: Union: float or (n,) shape numpy array. CE Loss
        """
        predict = mcl_repr(predict, squeeze=False, classes=self.classes)
        label_s = mcl_repr(label, squeeze=True, classes=self.classes)
        label = mcl_repr(label, squeeze=False, classes=self.classes)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Rows number mismatch: label {len(label)} and predict {len(predict)}')
        if self.class_weights:
            # 1-d weights
            weights = self._convert_label_to_class_weights(label_s)
            # 1-d
            losses = - weights * np.log(predict + 1e-15)[label.astype(bool)]
        else:
            # 1-d
            losses = - np.log(predict + 1e-15)[label.astype(bool)]

        if self.reduction == 'mean':
            return float(np.mean(losses.flatten()))
        elif self.reduction == 'sum':
            return float(np.sum(losses.flatten()))
        else:
            return losses.flatten()


class MSELoss(BaseLoss):
    def __init__(self, labels: np.ndarray = np.array([]), reduction: str = 'mean'):
        '''

        Args:
            labels: [Option] shape (n,) or (n,1) or (n,2), np.ndarray. Labels for computing class weight.
            reduction: [Option] str, e.g. 'mean', 'sum', 'none', else raise ValueError Exception.
                'mean': return mean of losses.
                'sum': return sum of losses.
                'none: return losses with same length with inputs.

        '''
        super(MSELoss, self).__init__(labels, reduction)

    def loss(self, predict: np.ndarray, label: np.ndarray) -> Union:
        """
        Compute MSE Loss

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.

        Returns:
            loss: Union: float or (n,) shape numpy array. MSE Loss
        """
        predict = bcl_repr(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        losses = (label - predict) ** 2
        if self.reduction == 'mean':
            return float(np.mean(losses.flatten()))
        elif self.reduction == 'sum':
            return float(np.sum(losses.flatten()))
        else:
            return losses.flatten()

    def gradient(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Compute gradients.

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.

        Returns:
            gradients: shape (n,) np.ndarray. Gradients with same length like input data.
        '''
        predict = bcl_repr(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        return 2 * (predict - label)

    def hessian(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Compute hessians.

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between 0 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are 0 or 1.abel:

        Returns:
            hessians: shape (n,) np.ndarray. Hessians with same length like input data.
        '''
        predict = bcl_repr(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        return np.full_like(label, 2)


class HingeLoss(ClassificationLoss):
    def __init__(self, labels: np.ndarray = np.array([]), reduction: str = 'mean'):
        '''

        Args:
            labels: [Option] shape (n,) or (n,1) or (n,2), np.ndarray. Labels for computing class weight.
            reduction: [Option] str, e.g. 'mean', 'sum', 'none', else raise ValueError Exception.
                'mean': return mean of losses.
                'sum': return sum of losses.
                'none: return losses with same length with inputs.

        '''
        super(HingeLoss, self).__init__(labels, reduction)

    def loss(self, predict: np.ndarray, label: np.ndarray) -> Union:
        """
        Compute Hinge Loss

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between -1 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are -1 or 1.

        Returns:
            loss: Union: float or (n,) shape numpy array. Hinge Loss
        """
        predict = bcl_repr(predict)
        label = bcl_repr(label)
        if len(predict) != len(label):
            raise ShapeMismatchError(f'Shape mismatch: label {len(label)} and predict {len(predict)}')
        if self.class_weights:
            weights = self._convert_label_to_class_weights(label)
            losses = weights * np.clip(1. - predict * label, 0.0, None)
        else:
            losses = np.clip(1. - predict * label, 0.0, None)
        if self.reduction == 'mean':
            return float(np.mean(losses.flatten()))
        elif self.reduction == 'sum':
            return float(np.sum(losses.flatten()))
        else:
            return losses.flatten()

    def gradient(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Compute gradients.

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between -1 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are -1 or 1.

        Returns:
            gradients: shape (n,) np.ndarray. Gradients with same length like input data.
        '''
        pass

    def hessian(self, predict: np.ndarray, label: np.ndarray) -> np.ndarray:
        '''
        Compute hessians.

        Args:
            predict: shape (n,) or (n,1) or (n,2) numpy array, values between -1 and 1.
            label: shape (n,) or (n,1) or (n,2) numpy array, values are -1 or 1.abel:

        Returns:
            hessians: shape (n,) np.ndarray. Hessians with same length like input data.
        '''
        pass


class RMSELoss(BaseLoss):
    def __init__(self):
        pass


    def loss(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSE Loss

        Args:
            predict: shape (n,) or (n,1) Torch.tensor, values between 0 and 5.
            label: shape (n,) or (n,1) Torch.tensor, values between 0 and 5.

        Returns:
            loss: (n,) shape Torch tensor. RMSE Loss
        """
        return torch.sum((predict-label)**2)


    def gradient(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def hessian(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_loss(loss_type: str, labels: np.ndarray = np.array([])):
    '''
    Generate Loss Object for loss type.

    Args:
        loss_type: str, the type of loss.
        labels: [Option] shape (n,) or (n,1) or (n,2), np.ndarray. Labels for computing class weight.

    Returns:
        Loss Object

    Examples:
    >>> loss_type = 'BCELoss'
    >>> loss = build_loss('BCELoss')
    >>> loss

    >>> #Output
    >>> <caffeine.utils.loss.BCELoss at 0x7f2d8b96fe48>
    Raises: LossTypeError
    '''
    if loss_type == Loss.BCELoss.value:
        return BCELoss(labels)
    elif loss_type == Loss.MSELoss.value:
        return MSELoss(labels)
    else:
        raise LossTypeError('Unsupported Loss Type')

