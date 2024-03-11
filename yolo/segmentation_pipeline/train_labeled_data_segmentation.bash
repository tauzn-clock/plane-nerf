#Get the directory the bash file is currently in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR


yolo task=segment \
mode=train \
model=yolov8m-seg.pt \
data=$DIR/../../data/jackal_floor_training_data_1/yolo/data.yaml \
epochs=100 \
imgsz=640