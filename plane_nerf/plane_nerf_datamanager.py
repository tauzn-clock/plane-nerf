"""
Template DataManager
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, List
from copy import deepcopy
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.model_components.ray_generators import RayGenerator

@dataclass
class PlaneNerfDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: PlaneNerfDataManager)


class PlaneNerfDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PlaneNerfDataManagerConfig

    def __init__(
        self,
        config: PlaneNerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_train_inerf(self, step: int) -> Tuple[RayBundle, Dict]:
        print("inerf")
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        print(image_batch["image"].shape)
        batch = self.train_pixel_sampler.sample(image_batch)
        print(batch["indices"].shape)
        ray_indices = batch["indices"]
        print(ray_indices[0])
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.train_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
    
    def setup_rays_inerf(self, RAYS = 4096, THRESHOLD = 40, KERNEL_SIZE = 5):
        image_batch = next(self.iter_train_image_dataloader)
        
        #Get image
        img = image_batch["image"][0] * image_batch["mask"][0]
        img *= 255.0
        img = img.cpu().numpy().astype(np.uint8)        
    
        #Get keypoints from image using SIFT
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(edgeThreshold=THRESHOLD)
        kp = sift.detect(gray, None)
        
        print("Number of keypoints: ", len(kp))
        
        #Get mask from keypoints
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for point in kp:
            x, y = point.pt
            x, y = int(x), int(y)
            mask[y-2:y+3, x-2:x+3] = 1
                
        #Dilate mask
        kernel = np.ones((KERNEL_SIZE,KERNEL_SIZE),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        
        #Get only masks that are in the original mask
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        mask = mask * image_batch["mask"][0].cpu().numpy().astype(np.uint8)
        
        print("Number of rays: ", mask.sum())
        
        (H, W,_) = img.shape
               
        if (mask.sum() >= RAYS):
            print("Reduce the number of rays")
            #Get random points from mask
            new_mask = mask.reshape(H*W)
            indices = np.where(new_mask == 1)[0]
            np.random.shuffle(indices)
            indices = indices[:RAYS]
            mask = np.zeros(H*W, dtype=np.uint8)
            mask[indices] = 1
            mask = mask.reshape((H, W, 1))
        else:
            print("Randomly select more rays")
            #Fill up mask with random points until you hit RAYS
            valid_indices =  image_batch["mask"][0].cpu().numpy().astype(np.uint8).reshape(H*W)
            valid_indices = np.where(valid_indices == 1)[0]
            np.random.shuffle(valid_indices)
            
            mask = mask.reshape(H*W)
            cnt = 0
            stop = int(RAYS - mask.sum())
            for idx in valid_indices:
                if (cnt == stop):
                    break
                if (mask[idx] == 0):
                    mask[idx] = 1
                    cnt += 1
            mask = mask.reshape((H, W, 1))

        print("Final number of rays: ", mask.sum())

        return img, mask
    
    KERNEL_SIZE = 5
    THRESHOLD = 40
        
    def get_inerf_raybundle_and_batch(self):
        #Only works for one image at a time
        batch = {}
        img, mask = self.setup_rays_inerf(RAYS=self.config.pixel_sampler.num_rays_per_batch, 
                                          THRESHOLD=self.THRESHOLD, 
                                          KERNEL_SIZE=self.KERNEL_SIZE)
        
        img_tensor = torch.tensor([])
        mask_tensor = torch.tensor([],dtype=torch.bool)
        indices_tensor = torch.tensor([],dtype=torch.int64)
        
        (H,W,_) = img.shape
        
        for i in range(H):
            for j in range(W):
                if (mask[i,j] == 1):
                    img_tensor = torch.cat((img_tensor, torch.tensor([img[i,j]/255.0])))
                    mask_tensor = torch.cat((mask_tensor, torch.tensor([[True]])))
                    indices_tensor = torch.cat((indices_tensor, torch.tensor([[0,i,j]])))
        
        batch["image"] = img_tensor
        batch["mask"] = mask_tensor
        batch["indices"] = indices_tensor
        
        print(batch)
        
        ray_bundle = self.train_ray_generator(indices_tensor)
        
        return ray_bundle, batch