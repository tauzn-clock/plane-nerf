"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from plane_nerf.plane_nerf_datamanager import (
    PlaneNerfDataManagerConfig,
)
from plane_nerf.plane_nerf_model import PlaneNerfConfig
from plane_nerf.plane_nerf_pipeline import (
    PlaneNerfPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from dataclasses import dataclass, field


plane_nerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="plane-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=10000,
        mixed_precision=True,
        pipeline=PlaneNerfPipelineConfig(
            datamanager=PlaneNerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    train_split_fraction = 1.0
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=PlaneNerfConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Plane Nerf Method",
)

plane_nerf_fast_method = MethodSpecification(
    config=TrainerConfig(
        method_name="plane-nerf-fast",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=10000,
        mixed_precision=True,
        pipeline=PlaneNerfPipelineConfig(
            datamanager=PlaneNerfDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    train_split_fraction = 1.0
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=PlaneNerfConfig(
                eval_num_rays_per_chunk=1 << 15,
                far_plane = 25.0,
                #hidden_dim = 32,
                #hidden_dim_color = 32,
                #hidden_dim_transient = 32,
                #num_proposal_samples_per_ray = (256*2, 96*2),
                #num_nerf_samples_per_ray = 48*2
                #max_res = 256,
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Plane Nerf Method (Faster)",
)