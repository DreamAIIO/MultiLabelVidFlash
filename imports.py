import flash
from flash.core.data.utils import download_data
from flash.video import VideoClassificationData, VideoClassifier
from pathlib import Path
import cv2
import os
import shutil
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from flash.video.classification.input_transform import VideoClassificationInputTransform
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo)
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, requires
from torchvision.transforms import Compose, CenterCrop
from torchvision.transforms import RandomCrop
from torch import nn
import kornia.augmentation as K

from torchvision import transforms as T
import torch.nn.functional as F

from flash.core.classification import ClassificationAdapterTask
from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DistributedSampler
from torchmetrics import Accuracy

import flash
from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.providers import _PYTORCHVIDEO
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE

from torchmetrics import Accuracy, F1Score, Metric