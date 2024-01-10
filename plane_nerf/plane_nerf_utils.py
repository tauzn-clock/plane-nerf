# Copyright 2023 Daoxin Zhong, University of Oxford. All rights reserved.
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

from typing import Dict, Tuple, Literal
from jaxtyping import Float 
from torch import Tensor
import torch


def invert_SO3xR3_pose(transform: Float[Tensor, "3 4"]) -> Float[Tensor, "3 4"]:
    """Inverts a 3x4 pose matrix.

    Args:
        transform: a 3x4 pose matrix

    Returns:
        The inverse of the pose matrix.
    """
    R = transform[:3, :3]
    t = transform[:3, 3]
    return torch.cat([R.t(), -R.t() @ t[:, None]], dim=1)

def transform_original_space_to_pose(
    poses: Float[Tensor, "num_poses 3 4"],
    applied_transform: Float[Tensor, "3 4"],
    applied_scale: float,
    camera_convention: Literal["opengl", "opencv"] = "opencv",
) -> Float[Tensor, "num_poses 3 4"]:
    """
    Transforms the original world coordinate system to the poses in the transformed space.
    Args:
        poses: Poses in the transformed space
        applied_transform: Transform matrix applied in the data processing step
        applied_scale: Scale used in the data processing step
        camera_convention: Camera system convention used for the transformed poses
    Returns:
        Original poses
    """
    output_poses = torch.cat(
        (
            poses,
            torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype, device=poses.device).repeat_interleave(len(poses), 0),
        ),
        1,
    )
    transform = torch.cat(
        (
            applied_transform,
            torch.tensor([[0, 0, 0, 1]], dtype=applied_transform.dtype, device=applied_transform.device),
        ),
        0,
    )
        
    output_poses = torch.einsum("ij,bjk->bik", transform, output_poses)
    
    output_poses[..., :3, 3] *= applied_scale
    
    if camera_convention == "opencv":
        raise ValueError(f"Camera convention {camera_convention} not yet supported.")
    elif camera_convention == "opengl":
        pass
    else:
        raise ValueError(f"Camera convention {camera_convention} is not supported.")
    return output_poses[:, :3]

