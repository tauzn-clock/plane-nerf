from typing import Dict, Tuple, Literal
from jaxtyping import Float 
from torch import Tensor
import torch
from torch import tensor
import numpy as np
from copy import deepcopy
from pathlib import Path
import os
import json
import cv2
from scipy.spatial.transform import Rotation 
from nerfstudio.data.dataparsers.base_dataparser import transform_poses_to_original_space
from plane_nerf.plane_nerf_optimizer import PlaneNerfCameraOptimizer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.camera_optimizers import CameraOptimizer

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


def correct_pose(given_pose, correction):
    """Correct the given pose by the correction.

    Args:
        given_pose: The given pose.
        correction: The correction.

    Returns:
        The corrected pose.
    """

    given_pose = torch.cat(
        (
            given_pose,
            torch.tensor([[[0, 0, 0, 1]]], dtype=given_pose.dtype, device=correction.device).repeat_interleave(len(given_pose), 0),
        ),
        1,
    )

    correction = torch.cat(
        (
            correction,
            torch.tensor([[[0, 0, 0, 1]]], dtype=correction.dtype, device=correction.device).repeat_interleave(len(correction), 0),
        ),
        1,
    )

    corrected_pose = torch.matmul(correction,given_pose)

    return corrected_pose[:, :3, :4]

def get_corrected_pose(trainer):
    """Get the corrected pose.

    Args:
        trainer: The trainer.

    Returns:
        The corrected pose.
    """

    camera = trainer.pipeline.datamanager.train_dataparser_outputs.cameras.camera_to_worlds.to(trainer.pipeline.device)
    #print("camera", camera)

    correction = trainer.pipeline.model.camera_optimizer.get_correction_matrices()

    R = correction[:, :3, :3]
    t = correction[:, :3, 3]

    corrected_trans = camera[:, :3, 3] + t
    corrected_trans = corrected_trans.reshape(-1, 3, 1)
    #print(corrected_trans)

    corrected_rot = torch.bmm(R, camera[:, :3, :3])
    #print(corrected_rot)

    corrected_pose = torch.cat((corrected_rot, corrected_trans), 2)

    corrected_pose = transform_poses_to_original_space(
        corrected_pose,
        trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_transform.to(trainer.pipeline.device),
        trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
        "opengl"
    )

    return corrected_pose

def load_eval_image_into_pipeline(pipeline, eval_path, transforms, starting_pose=None):
    
    data = transforms["frames"]
        
    custom_train_dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    custom_train_dataparser_outputs.image_filenames = []
    custom_train_dataparser_outputs.mask_filenames = []
    custom_train_dataparser_outputs.mask_midpt = []
    
    camera_to_worlds = tensor([]).float()
    fx = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.fx[0]]*len(data),0)
    fy = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.fy[0]]*len(data),0)
    cx = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.cx[0]]*len(data),0)
    cy = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.cy[0]]*len(data),0)
    distortion_params = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.distortion_params[0]]*len(data),0)
    height = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.height[0]]*len(data),0)
    width = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.width[0]]*len(data),0)
    camera_type = torch.stack([pipeline.datamanager.train_dataparser_outputs.cameras.camera_type[0]]*len(data),0)
    
    for i in range(len(data)):
        custom_train_dataparser_outputs.image_filenames.append(Path(os.path.join(eval_path,data[i]["file_path"])).as_posix())
        custom_train_dataparser_outputs.mask_filenames.append(Path(os.path.join(eval_path,data[i]["mask_path"])).as_posix())
        mask = cv2.imread(custom_train_dataparser_outputs.mask_filenames[-1], cv2.IMREAD_GRAYSCALE)
        custom_train_dataparser_outputs.mask_midpt.append(get_mask_midpt(mask))
        if starting_pose is None:
            tf = np.asarray(data[i]["transform_matrix"])
        else:
            tf = starting_pose[i]
        tf = tf[:3, :]
        camera_to_worlds = torch.cat([camera_to_worlds, tensor([tf]).float()], 0)   
    
    custom_train_dataparser_outputs.mask_midpt = torch.tensor(custom_train_dataparser_outputs.mask_midpt).float().to(pipeline.device)

    custom_cameras = pipeline.datamanager.train_dataparser_outputs.cameras
    custom_cameras.camera_to_worlds = transform_original_space_to_pose(camera_to_worlds,
                                                                        pipeline.datamanager.train_dataparser_outputs.dataparser_transform,
                                                                        pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
                                                                        "opengl")
    custom_cameras.fx = fx
    custom_cameras.fy = fy
    custom_cameras.cx = cx
    custom_cameras.cy = cy
    custom_cameras.distortion_params = distortion_params
    custom_cameras.height = height
    custom_cameras.width = width
    custom_cameras.camera_type = camera_type
    custom_train_dataparser_outputs.cameras = custom_cameras
        
    pipeline.datamanager.train_dataparser_outputs = custom_train_dataparser_outputs
    pipeline.datamanager.train_dataset = pipeline.datamanager.create_train_dataset()
    pipeline.datamanager.setup_train()
    
    
    return pipeline


def get_relative_pose(ground_truth_poses, target_poses):
    """Get the relative pose.

    Args:
        ground_truth_poses: The ground truth poses.
        target_poses: The target poses.

    Returns:
        The relative pose.
    """
    
    dtype = ground_truth_poses.dtype
    device = ground_truth_poses.device
    
    ground_truth_4x4 = torch.cat(
        (
            ground_truth_poses,
            torch.tensor([[[0, 0, 0, 1]]], dtype = dtype, device = device).repeat_interleave(len(ground_truth_poses), 0),
        ),
        1,
    )

    R_inv = target_poses[:, :3, :3].transpose(1,2)
    t_inv = torch.matmul(target_poses[:, :3, :3].transpose(1,2), -target_poses[:, :3, 3].unsqueeze(-1))
    # Concat R_inv and t_inv   
    target_4x4 = torch.cat([R_inv, t_inv], dim=2)

    target_4x4 = torch.cat(
        (
            target_4x4,
            torch.tensor([[[0, 0, 0, 1]]], dtype = dtype, device = device).repeat_interleave(len(target_poses), 0),
        ),
        1, 
    )
    
    relative_pose = torch.matmul(ground_truth_4x4, target_4x4)
    
    return relative_pose

def get_absolute_diff_for_pose(pose):
    """Get the absolute difference for the pose.

    Args:
        pose: The pose.

    Returns:
        The absolute difference for the pose.
    """
    translation = pose[:, :3, 3]
    rotation = pose[:, :3, :3]
    
    translation_diff = torch.norm(translation, dim=1)

    trace = rotation.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    angle_rot = torch.acos((trace- 1) / 2)

    return translation_diff, angle_rot

def get_image(pipeline, pose):
    camera = pipeline.datamanager.train_dataparser_outputs.cameras
    camera.camera_to_worlds = transform_original_space_to_pose(pose,
                                            pipeline.datamanager.train_dataparser_outputs.dataparser_transform.to(pipeline.device),
                                            pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
                                            "opengl").to("cpu")
    outputs = pipeline.model.get_outputs_for_camera(camera=camera)    
    return outputs
                                           
def get_mask_midpt(mask):
    """Get the mask mid point of a numpy mask.

    Args:
        mask: The mask.

    Returns:
        The mask mid point.
    """
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if len(contours) > 0:
        # Find the centroid of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        # Calculate the centroid coordinates
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None

def get_camera_intrinsic(camera):
    intrinsic = torch.zeros((3,4))
    
    intrinsic[0,0] = camera.fx[0]
    intrinsic[0,2] = camera.cx[0]
    intrinsic[1,1] = camera.fy[0]
    intrinsic[1,2] = camera.cy[0]
    intrinsic[2,2] = 1

    return intrinsic

def get_origin(pose, intrinsic):
    """Get the pixel coordinate of a origin.

    Args:
        pose: The pose.
        camera: Camera intrinsicsalpha = 0.5
    """
    transform = pose
    inv_transform = torch.linalg.inv(transform.float())

    #Rotate about x axis by 180 degrees
    rot = torch.eye(4).to(pose.device)
    rot[1,1] = -1
    rot[2,2] = -1
    inv_transform = torch.matmul(rot, inv_transform)

    plane_index = torch.matmul(intrinsic, inv_transform)
    plane_index = plane_index[:,:,3]
    origin_coord = plane_index[:,:2]/plane_index[:,2]
    return origin_coord

def inerf(trainer, ITERATION = 1000, LR = 1e-3, GROUND_TRUTH_POSE=None):
    trainer.pipeline.datamanager.get_inerf_batch()  
    trainer.pipeline.datamanager.inerf_batch["image"] = trainer.pipeline.datamanager.inerf_batch["image"].to(trainer.pipeline.device)
    trainer.pipeline.train()

    if GROUND_TRUTH_POSE is not None:
        store = torch.tensor([])

    for i in range(ITERATION):
        if (GROUND_TRUTH_POSE is not None):
            corrected_pose = get_corrected_pose(trainer)
            relative_pose = get_relative_pose(GROUND_TRUTH_POSE, corrected_pose)
            t_diff, r_diff = get_absolute_diff_for_pose(relative_pose)
            store = torch.cat((store, torch.tensor([[i+1, torch.mean(t_diff), torch.mean(r_diff)]])), 0)
            #Get averrage absolute translation and rotation error
            #print("Average translation error: ", torch.mean(t_diff))
            #print("Average rotation error: ", torch.mean(r_diff))
            #print(loss)
        loss = trainer.train_iteration_inerf(optimizer_lr = LR)

    corrected_pose = get_corrected_pose(trainer)
    ans = {}
    ans["corrected_pose"] = corrected_pose
    ans["loss"] = loss
    if (GROUND_TRUTH_POSE is not None):
        ans["store"] = store
        ans["relative_pose"] = relative_pose
        ans["translation_diff"] = torch.mean(t_diff)
        ans["rotation_diff"] = torch.mean(r_diff)
    
    return ans

def get_intrinsic(pipeline,idx):

    intrinsic = torch.zeros((3,4))
    intrinsic[0,0] = pipeline.datamanager.train_dataparser_outputs.cameras.fx[idx]
    intrinsic[0,2] = pipeline.datamanager.train_dataparser_outputs.cameras.cx[idx]
    intrinsic[1,1] = pipeline.datamanager.train_dataparser_outputs.cameras.fy[idx]
    intrinsic[1,2] = pipeline.datamanager.train_dataparser_outputs.cameras.cy[idx]
    intrinsic[2,2] = 1
    
    return intrinsic.to(pipeline.device)

def get_extrinsic(camera,idx,pipeline):
    extrinsic = camera.camera_to_worlds[idx]
    extrinsic = extrinsic.reshape(1,3,4).to(pipeline.device)
    extrinsic = transform_poses_to_original_space(extrinsic,
                                                  pipeline.datamanager.train_dataparser_outputs.dataparser_transform.to(pipeline.device),
                                                  pipeline.datamanager.train_dataparser_outputs.dataparser_scale,
                                                  "opengl"
                                                  )
    
    extrinsic = extrinsic.reshape(3,4)
    extrinsic = torch.concat((extrinsic, torch.tensor([0,0,0,1]).reshape(1,4).to(pipeline.device)),0)

    return extrinsic

def get_footprint(intrinsic, extrinsic, mask):
    assert intrinsic.device == extrinsic.device
    device = intrinsic.device

    matrix = torch.matmul(intrinsic, extrinsic)
    matrix = torch.concat((matrix[:,:2], matrix[:,3].reshape(3,1)),1)
    matrix = torch.inverse(matrix)

    #Get contour of mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    #Get footprint
    footprint = torch.tensor([]).to(device)
    for coord in largest_contour:
        (x,y) = coord[0]
        point = torch.tensor([x,y,1]).reshape(3,1).float().to(device)
        point = torch.matmul(matrix, point)
        point = point/point[2]
        point = point.to(device)
        footprint = torch.cat((footprint, point[:2]), 1)

    return footprint

def get_image_with_footprint(pipeline, camera, intrinsic, extrinsic, footprint):
    outputs = pipeline.model.get_outputs_for_camera(camera=camera)  

    output_image = outputs["rgb"].reshape(camera.height, camera.width, 3).cpu().numpy()
    output_image = (output_image * 255).astype(np.uint8)
    pixel_coord = []
    for i in range(footprint.shape[1]):
        (x,y) = footprint[:,i]
        footprint_pose = torch.tensor([x,y,0,1]).reshape(4,1).to(pipeline.device)

        pixel = intrinsic @ extrinsic @ footprint_pose
        pixel = pixel / pixel[2]

        pixel_coord.append((int(pixel[0]), int(pixel[1])))
    
    output_image = cv2.polylines(output_image, [np.array(pixel_coord)], True, (0, 255, 0), 2)

    #Draw orgin
    origin_pose = torch.tensor([0,0,0,1]).reshape(4,1).float().to(pipeline.device)
    pixel = intrinsic @ extrinsic @ origin_pose
    pixel = pixel / pixel[2]
    output_image = cv2.circle(output_image, (int(pixel[0]), int(pixel[1])), 5, (255, 0, 0), -1)

    return output_image