"""losses."""

from .qr import QRLoss, RQLoss
from .loss import DiceLoss, CE_DiceLoss

__all__ = ('QRLoss', 'RQLoss','DiceLoss', 'CE_DiceLoss')