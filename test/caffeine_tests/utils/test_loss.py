import numpy as np
import os
from logging import getLogger

from caffeine.utils.loss import BCELoss, MSELoss


def test_loss():
    logger = getLogger(__name__)
    reductions = ['none', 'mean', 'sum']
    label = np.array([[0], [0], [0], [1], [1]])
    label1 = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])

    preds_very_good = np.array([[0.01], [0.01], [0.01], [0.99], [0.99]])
    preds_good = np.array([[0.1], [0.3], [0.1], [0.9], [0.7]])
    preds_middle = np.array([[0.6], [0.3], [0.3], [0.9], [0.7]])
    preds_bad = np.array([[0.7], [0.5], [0.8], [0.1], [0.3]])
    preds_bad1 = np.array([[0.3, 0.7], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1], [0.7, 0.3]])

    for reduction in reductions:
        # ==================init losses================================
        bce_loss = BCELoss(reduction=reduction)
        bce_weight_loss = BCELoss(label, reduction=reduction)
        mse_loss = MSELoss(reduction=reduction)
        mse_weight_loss = MSELoss(reduction=reduction)
        losses = [bce_loss, bce_weight_loss, mse_loss, mse_weight_loss]
        # ==================test losses================================
        for loss in losses:  # todo 这里还没有验证gradient和hessian算的对不对呢。虽然我觉得很对
            logger.info(f'Start test {loss.__class__.__name__} with params {loss.__dict__}')
            very_good = loss.loss(preds_very_good, label)
            good = loss.loss(preds_good, label)
            middle = loss.loss(preds_middle, label)
            bad = loss.loss(preds_bad.T, label)
            logger.info(f'very good loss is {very_good}')
            logger.info(f'good loss is {good}')
            logger.info(f'middle loss is {middle}')
            logger.info(f'bad loss is {bad}')
            gradient_very_good = loss.gradient(preds_very_good, label)
            hessian_very_good = loss.hessian(preds_very_good, label)
            # gradient_good = loss.gradient(preds_good, label)
            # gradient_middle = loss.gradient(preds_middle, label)
            # graidnet_bad = loss.gradient(preds_bad.T, label)
            logger.info(f'gradient very good shape is {gradient_very_good.shape}')
            logger.info(f'gradient very good is {gradient_very_good}')
            logger.info(f'hessian very good shape is {hessian_very_good}')
            logger.info(f'hessian very good is {hessian_very_good}')
            if reduction == 'none':
                assert (loss.loss(preds_very_good, label) == loss.loss(preds_very_good, label1)).all()
                assert (loss.loss(preds_bad, label) == loss.loss(preds_bad1, label)).all()
                assert (loss.loss(preds_bad, label1) == loss.loss(preds_bad1, label)).all()
                assert (loss.loss(preds_bad, label) == loss.loss(preds_bad1, label1)).all()
            else:
                assert very_good < good and good < middle and middle < bad
                assert loss.loss(preds_very_good, label) == loss.loss(preds_very_good, label1)
                assert loss.loss(preds_bad, label) == loss.loss(preds_bad1, label)
                assert loss.loss(preds_bad, label1) == loss.loss(preds_bad1, label)
                assert loss.loss(preds_bad, label) == loss.loss(preds_bad1, label1)
            assert (loss.gradient(preds_very_good, label) == loss.gradient(preds_very_good, label1)).all()
            assert (loss.gradient(preds_bad, label) == loss.gradient(preds_bad1, label)).all()
            assert (loss.gradient(preds_bad, label1) == loss.gradient(preds_bad1, label)).all()
            assert (loss.gradient(preds_bad, label) == loss.gradient(preds_bad1, label1)).all()
            logger.info(f'Complete test {loss.__class__.__name__} with params {loss.__dict__}')


if __name__ == '__main__':
    test_loss()
