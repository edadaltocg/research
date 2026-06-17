#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

as_ci_user() {
  # NB: Pass on PATH and LD_LIBRARY_PATH to sudo invocation
  # NB: This must be run from a directory that the user has access to
  sudo -E -H -u ci-user env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=${PATH}" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "$@"
}
