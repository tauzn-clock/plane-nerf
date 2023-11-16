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

from nerfstudio.data.scene_box import SceneBox
from jaxtyping import Float
from torch import Tensor

class PlaneSceneBox(SceneBox):
    @staticmethod
    def get_normalized_positions(positions: Float[Tensor, "*batch 3"], aabb: Float[Tensor, "2 3"]):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = aabb[1] - aabb[0]
        normalized_positions = (positions - aabb[0]) / aabb_lengths

        #Set the z coordinate to 0.5
        normalized_positions = normalized_positions.at[:, 2].set(0.5)

        return normalized_positions
