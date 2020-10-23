#!/usr/bin/env bash

n=3 # num layers
working_dir=""
data_dir=$working_dir"/Data/camembert"
embed="base_ccnet"
train=$data_dir"/"$embed"/camembert_train_wmanuel_lref.u8.txt"
dev=$data_dir"/"$embed"/camembert_dev_wmanuel_lref.u8.txt"
test_=$data_dir"/"$embed"/camembert_test_wmanuel_lref.u8.txt"
res_dir=$working_dir"/results/MEDIA/camembert"$embed
model_name=$trained"_"$embed

echo $res_dir


if [ ! -d $res_dir ]; then
   mkdir -p $res_dir
   mkdir $res_dir"/models"
   mkdir $res_dir"/log"
   mkdir $res_dir"/predictions"
fi



for m in LSTM; do
  for b in {16,32,64,128}; do
    for h in {128,256,512}; do

	 python SLU_BLSTM_freeze.py --mode $m --num_epochs 200 --batch_size $b --hidden_size $h --num_layers $n\
	 --char_dim 30 --num_filters 30 --tag_space 0 \
 	--learning_rate 0.001 --decay_rate 0.05 --schedule 1 --gamma 0.0 --task MEDIA --optim ADAM \
	 --dropout std --p 0.5 --unk_replace 0.0 --bidirectional True\
 	--data_path $res_dir --modelname $model_name  --train $train  --dev $dev  --test $test_
done
done
done
