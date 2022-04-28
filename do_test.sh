collection=$1
visual_feature=$2
root_path=$3
model_dir=$4

# training

python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir