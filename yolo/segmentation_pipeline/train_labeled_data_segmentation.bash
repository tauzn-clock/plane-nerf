#Get the directory the bash file is currently in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
data_dir=$DIR'/../../data/jackal_floor_training_data_1/yolo'

cd $data_dir

yolo task=segment \
mode=train \
model=yolov8m-seg.pt \
data=$data_dir'/data.yaml' \
epochs=100 \
imgsz=640 \
name=model_$current_time