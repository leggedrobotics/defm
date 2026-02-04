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
    __version__ = "1.0.1"
