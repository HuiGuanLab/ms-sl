collection=activitynet
visual_feature=i3d

map_size=32
model_name=MS_SL_Net
exp_id=$1
root_path=$2
device_ids=$3
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --map_size $map_size --model_name $model_name --device_ids $device_ids