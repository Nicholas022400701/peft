# Copyright 2026-present the HuggingFace Inc. team.
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

import copy

import pytest
import torch
from torch import nn

from peft import OFTConfig, get_peft_model
from peft.utils import infer_device


class ConvModel(nn.Module):
    def __init__(self, **conv_kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(4, 6, **conv_kwargs)

    def forward(self, X):
        return self.conv2d(X)


# inputs larger than the kernel, so that the unfolded patches overlap
CONV_KWARGS = [
    {"kernel_size": 3, "padding": 1},
    {"kernel_size": (3, 5), "padding": (1, 2)},
    {"kernel_size": 3, "stride": (1, 2), "padding": (1, 0)},
    {"kernel_size": 3, "padding": 1, "groups": 2},
    {"kernel_size": 3, "padding": 1, "padding_mode": "reflect"},
]


class TestOft:
    device = infer_device()

    def get_model_and_input(self, **conv_kwargs):
        torch.manual_seed(0)
        model = ConvModel(**conv_kwargs).to(self.device).eval()
        X = torch.randn(2, 4, 8, 8, device=self.device)
        return model, X

    @pytest.mark.parametrize("conv_kwargs", CONV_KWARGS)
    def test_oft_conv2d_identity_init_matches_base_model(self, conv_kwargs):
        model, X = self.get_model_and_input(**conv_kwargs)
        output_base = model(X)

        config = OFTConfig(r=2, oft_block_size=0, target_modules=["conv2d"])
        peft_model = get_peft_model(copy.deepcopy(model), config).eval()
        output_peft = peft_model(X)

        assert output_peft.shape == output_base.shape
        assert torch.allclose(output_peft, output_base, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("conv_kwargs", CONV_KWARGS)
    @pytest.mark.parametrize("coft", [False, True])
    def test_oft_conv2d_forward_matches_merged_model(self, conv_kwargs, coft):
        model, X = self.get_model_and_input(**conv_kwargs)

        config = OFTConfig(r=2, oft_block_size=0, target_modules=["conv2d"], init_weights=False, coft=coft)
        peft_model = get_peft_model(model, config).eval()
        output_unmerged = peft_model(X)

        peft_model.merge_adapter()
        output_merged = peft_model(X)
        assert torch.allclose(output_unmerged, output_merged, atol=1e-6, rtol=1e-6)

        peft_model.unmerge_adapter()
        output_unmerged_again = peft_model(X)
        assert torch.allclose(output_unmerged, output_unmerged_again, atol=1e-5, rtol=1e-5)
