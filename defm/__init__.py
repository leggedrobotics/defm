# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Expose package version: prefer installed metadata, fall back to a default.
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    # Python <3.8 fallback (unlikely here)
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("defm")
except PackageNotFoundError:
    # fallback for source checkout / editable installs — keep in sync with pyproject.toml
    __version__ = "1.1.0"

# Preprocessing helpers at the top level, as used in the README (`from defm import preprocess_depth_image`)
from .utils import (  # noqa: E402,F401
    convert_to_3channel_metric,
    preprocess_depth_batch,
    preprocess_depth_dav2,
    preprocess_depth_image,
)
