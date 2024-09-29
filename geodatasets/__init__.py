# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo geodatasets."""

from .errors import DatasetNotFoundError, DependencyNotFoundError, RGBBandsMissingError
from .geo import (
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    UnionDataset,
    VectorDataset,
)
from .splits import (
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)
from .utils import (
    BoundingBox,
    concat_samples,
    merge_samples,
    stack_samples,
    unbind_samples,
)

__all__ = (
    # Base classes
    'GeoDataset',
    'IntersectionDataset',
    'NonGeoClassificationDataset',
    'NonGeoDataset',
    'RasterDataset',
    'UnionDataset',
    'VectorDataset',
    # Utilities
    'BoundingBox',
    'concat_samples',
    'merge_samples',
    'stack_samples',
    'unbind_samples',
    # Splits
    'random_bbox_assignment',
    'random_bbox_splitting',
    'random_grid_cell_assignment',
    'roi_split',
    'time_series_split',
    # Errors
    'DatasetNotFoundError',
    'DependencyNotFoundError',
    'RGBBandsMissingError',
)