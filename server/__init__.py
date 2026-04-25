# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cascade Guard environment server components."""
# AFTER
try:
    from .cascade_environment import CascadeEnvironment
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cascade_environment import CascadeEnvironment

__all__ = ["CascadeEnvironment"]
