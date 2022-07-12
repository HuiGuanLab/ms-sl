collection=charades
visual_feature=i3d_rgb_lgi
clip_scale_w=0.5
frame_scale_w=0.5
exp_id=$1
root_path=$2
device_ids=$3
# training

python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --clip_scale_w $clip_scale_w --frame_scale_w $frame_scale_w \
                    --device_ids $device_ids