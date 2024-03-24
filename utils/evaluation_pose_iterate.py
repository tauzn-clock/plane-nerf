import os
import json
import torch
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output
from nerfstudio.utils.eval_utils import eval_setup
from plane_nerf.inerf_trainer import load_data_into_trainer
from plane_nerf.inerf_utils import load_eval_image_into_pipeline, inerf, inerf_v2

os.chdir('/workspace/plane-nerf')
MODEL_PATH = "/workspace/plane-nerf/outputs/jackal_floor_training_data_1/plane-nerf/2024-03-11_145657"
DATA_PATH = "/workspace/plane-nerf/data/jackal_floor_evaluation_data"
GROUND_TRUTH_PATH = os.path.join(DATA_PATH, "ground_truth.json")
SAVE_FILE_NAME = "eval_results_"+str(time.strftime("%Y-%m-%d_%H%M%S"))+".csv"

config_path = os.path.join(MODEL_PATH, "config.yml")
config, pipeline, _, _ = eval_setup(
                        Path(config_path),
                        test_mode="inference",
                    )
transform_file_path = "transforms_10.json"
with open(os.path.join(DATA_PATH, transform_file_path)) as f:
    transform = json.load(f)

with open(GROUND_TRUTH_PATH) as f:
    ground_truth = json.load(f)

output = []

for f in range(len(transform["frames"])):
    transform_dup = transform.copy()
    transform_dup["frames"] = [transform["frames"][f]]
        
    pipeline = load_eval_image_into_pipeline(pipeline,DATA_PATH,transform_dup)

    config.pipeline.datamanager.pixel_sampler.num_rays_per_batch = 4096 

    trainer = load_data_into_trainer(
        config,
        pipeline,
        plane_optimizer = True
    )
    trainer.pipeline.datamanager.KERNEL_SIZE = 5
    trainer.pipeline.datamanager.THRESHOLD = 40
    trainer.pipeline.datamanager.METHOD = "sift"

    clear_output(wait=True)
    print(f)

    tf = ground_truth["frames"][f]["transform_matrix"]
    tf = np.asarray(tf)
    tf = tf[:3, :4 ]
    ground_truth_poses = [tf]
    ground_truth_poses = torch.tensor(ground_truth_poses).to(pipeline.device)
    
    ans = inerf(trainer, ITERATION=500, LR = 1e-2, GROUND_TRUTH_POSE=ground_truth_poses)
    #ans = inerf_v2(trainer, GROUND_TRUTH_POSE=ground_truth_poses)
    output.append(ans["store"].detach().cpu().numpy().flatten())
    np.savetxt(os.path.join(MODEL_PATH,SAVE_FILE_NAME), np.asarray(output), delimiter=",")

    torch.cuda.empty_cache()