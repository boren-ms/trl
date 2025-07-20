# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""Job utilities for TRL scripts."""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytz
import shortuuid


def uuid4():
    short_id = shortuuid.ShortUUID().random(length=4)
    return short_id


def is_master():
    """Check if the current process is the master process."""
    local_rank = os.environ.get("LOCAL_RANK", "0")
    rank = os.environ.get("RANK", "0")
    print("LocalRank:", local_rank)
    print("Rank:", rank)
    return local_rank == "0" and rank == "0"


def get_job_name(jobname=None):
    """Get a unique job name."""
    if jobname:
        return jobname
    if "--config" in sys.argv:
        # use config file name as job name
        config_file = sys.argv[sys.argv.index("--config") + 1]
        jobname = Path(config_file).stem.split(".")[0]
        return jobname
    # use current time as job name
    tz = pytz.timezone("America/Los_Angeles")  # UTC-7/UTC-8 depending on DST
    return datetime.now(tz).strftime("%Y%m%d-%H%M%S")