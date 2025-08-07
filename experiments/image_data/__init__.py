# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading."""
# pylint:disable=g-multiple-import
# pylint:disable=g-importing-member

from experiments.image_data import chexpert, mimic_cxr
from experiments.image_data.augmult import AugmultConfig
from experiments.image_data.base import (
    DataInputs,
    DatasetConfig,
    ImageDatasetConfig,
)
from experiments.image_data.chexpert import (
    AbstractChexpertLoader,
    ChexpertTestInternalConfig,
    ChexpertTestOfficialConfig,
    ChexpertTrainInternalConfig,
    ChexpertTrainOfficialConfig,
    ChexpertValidInternalConfig,
    ChexpertValidOfficialConfig,
)
from experiments.image_data.imagenet import (
    ImageNetConfig,
    ImageNetLoader,
    ImageNetNumSamples,
    ImagenetTestConfig,
    ImagenetTrainConfig,
    ImagenetTrainValidConfig,
    ImagenetValidConfig,
)
from experiments.image_data.loader import (
    DataLoader,
    LoadDatasetFn,
    LoadRawDataTfFn,
    default_load_dataset,
    grain_load_dataset,
)
from experiments.image_data.mimic_cxr import (
    AbstractMimicCxrLoader,
    MimicCxrTestInternalConfig,
    MimicCxrTestOfficialConfig,
    MimicCxrTrainInternalConfig,
    MimicCxrTrainOfficialConfig,
    MimicCxrValidInternalConfig,
    MimicCxrValidOfficialConfig,
)
from experiments.image_data.mnist_cifar_svhn import (
    Cifar10Loader,
    Cifar10TestConfig,
    Cifar10TrainConfig,
    Cifar10TrainValidConfig,
    Cifar10ValidConfig,
    Cifar100Loader,
    Cifar100TestConfig,
    Cifar100TrainConfig,
    Cifar100TrainValidConfig,
    Cifar100ValidConfig,
    MnistLoader,
    MnistTestConfig,
    MnistTrainConfig,
    MnistTrainValidConfig,
    MnistValidConfig,
    SvhnLoader,
    SvhnTestConfig,
    SvhnTrainConfig,
    SvhnTrainValidConfig,
    SvhnValidConfig,
)
from experiments.image_data.places365 import (
    Places365Loader,
    Places365NumSamples,
    Places365Testconfig,
    Places365TrainConfig,
    Places365TrainValidConfig,
    Places365ValidConfig,
)

MULTILABEL_DATASETS = (
    chexpert.ChexpertConfig,
    mimic_cxr.MimicCxrConfig,
)
