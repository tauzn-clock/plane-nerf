# The project folder must contain a folder "images" with all the images.
echo "Where is this script?"
SCRIPT_PATH=$(dirname $(realpath $0))
echo SCRIPT_PATH=$SCRIPT_PATH

DATASET_NAME=gerrard-hall
DATASET_PATH=$SCRIPT_PATH/../data/$DATASET_NAME
echo "Where is the dataset?"
echo DATASET_PATH=$DATASET_PATH

#Delete database.db if it exists
if [ -f $DATASET_PATH/database.db ]; then
  rm $DATASET_PATH/database.db
fi

# Create the database
colmap database_creator \
  --database_path $DATASET_PATH/database.db

colmap feature_extractor \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
#  --ImageReader.mask_path $DATASET_PATH/masks \

colmap exhaustive_matcher \
  --database_path $DATASET_PATH/database.db \

mkdir -p $DATASET_PATH/sparse

colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH/images \
  --output_path $DATASET_PATH/sparse \
  --Mapper.ba_global_use_pba 1 \
  --Mapper.ba_global_pba_gpu_index 0 \
  --Mapper.min_num_matches 15 \
  --Mapper.init_min_num_inliers 15 \
  --Mapper.abs_pose_max_error 4.0 \
  --Mapper.abs_pose_min_num_inliers 10 \
  --Mapper.filter_max_reproj_error 4.0 \
  --Mapper.max_reg_trials 2

mkdir -p $DATASET_PATH/dense

colmap image_undistorter \
  --image_path $DATASET_PATH/images \
  --input_path $DATASET_PATH/sparse/0 \
  --output_path $DATASET_PATH/dense \
  --output_type COLMAP \
  --max_image_size 2000

colmap patch_match_stereo \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path $DATASET_PATH/dense \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path $DATASET_PATH/dense/fused.ply

colmap poisson_mesher \
  --input_path $DATASET_PATH/dense/fused.ply \
  --output_path $DATASET_PATH/dense/meshed-poisson.ply

colmap delaunay_mesher \
  --input_path $DATASET_PATH/dense/fused.ply \
  --output_path $DATASET_PATH/dense/meshed-delaunay.ply