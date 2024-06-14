import dataclasses
import functools
import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast, DefaultDict
from jaxtyping import Float 
from collections import defaultdict
import torch
from torch import Tensor
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler
from torchmetrics.regression import MeanSquaredError

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from plane_nerf.inerf_utils import get_corrected_pose, load_eval_image_into_pipeline, get_camera_intrinsic, get_origin
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from plane_nerf.plane_nerf_optimizer import PlaneNerfCameraOptimizer

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str

class INerfTrainer(Trainer):
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]
    
    def setup_inerf(self, pipeline ):        
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        
        self.pipeline = pipeline
        self.optimizers = self.setup_optimizers()

        self._load_checkpoint_inerf()
        self.camera_intrinsic = get_camera_intrinsic(self.pipeline.datamanager.train_dataparser_outputs.cameras).to(pipeline.device)
        self.rgb_loss_func = MeanSquaredError().to(pipeline.device)
        #self.rgb_loss_func = torch.nn.L1Loss().to(pipeline.device)
        
    def _load_checkpoint_inerf(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self.loaded_state = loaded_state

            loaded_state["step"] = 0
            loaded_state["optimizers"].pop("camera_opt", None)
            loaded_state["pipeline"].pop("_model.camera_optimizer.pose_adjustment", None)

            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    def get_loss(self):
        batch = self.pipeline.datamanager.inerf_batch
        gt_rgb = batch["image"]  # RGB or RGBA image
        ray_bundle = self.pipeline.datamanager.train_ray_generator(batch["indices"])
        predicted_rgb = self.pipeline.model.get_rgb_outputs(ray_bundle=ray_bundle)
        
        rgb_loss = self.rgb_loss_func(gt_rgb, predicted_rgb)

        #Pixel Loss
        mask_centre = self.pipeline.datamanager.train_dataparser_outputs.mask_midpt
        corrected_pose = get_corrected_pose(self)
        corrected_pose = torch.cat(
            (
                corrected_pose,
                torch.tensor([[[0, 0, 0, 1]]], dtype=corrected_pose.dtype, device=corrected_pose.device).repeat_interleave(len(corrected_pose), 0),
            ),
            1,
        )
        expected_origin = get_origin(corrected_pose,self.camera_intrinsic)
        pixel_loss = torch.square(torch.norm(expected_origin - mask_centre)) 
        pixel_loss = torch.max(pixel_loss, torch.tensor([400], dtype=pixel_loss.dtype, device=pixel_loss.device))
        pixel_loss = pixel_loss * 1e-5

        return {"loss": rgb_loss + pixel_loss}

    @profiler.time_function
    def train_iteration_inerf(self, optimizer_lr: Optional[Float] = None) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_some(["camera_opt"])

        #cpu_or_cuda_str: str = self.device.split(":")[0]
        #cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str
        #with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
        loss = self.get_loss()
        self.grad_scaler.scale(loss["loss"]).backward()  # type: ignore

        if optimizer_lr is not None:
            self.optimizers.optimizers["camera_opt"].param_groups[0]['lr'] = optimizer_lr
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, ["camera_opt"])

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        # if scale <= self.grad_scaler.get_scale():
        #     self.optimizers.scheduler_step_all(step)

        return loss
def load_data_into_trainer(
    config,
    pipeline,
    plane_optimizer = True
):

    if plane_optimizer:
        print("Loading PlaneNerfCameraOptimizer")
        custom_camera_optimizer = PlaneNerfCameraOptimizer(
            config = pipeline.model.camera_optimizer.config,
            num_cameras = len(pipeline.datamanager.train_dataset),
            device = pipeline.device,
        )
    else:
        print("Loading CameraOptimizer")
        custom_camera_optimizer = CameraOptimizer(
            config = pipeline.model.camera_optimizer.config,
            num_cameras = len(pipeline.datamanager.train_dataset),
            device = pipeline.device,
        )
        
    custom_camera_optimizer.config.rot_l2_penalty = 0 #
    custom_camera_optimizer.config.trans_l2_penalty = 0 #
    pipeline.model.camera_optimizer = custom_camera_optimizer
    trainer = INerfTrainer(config)
    trainer.setup_inerf(pipeline)
    
    return trainer

