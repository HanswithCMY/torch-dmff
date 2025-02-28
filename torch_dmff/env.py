# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import torch

LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)
if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)
