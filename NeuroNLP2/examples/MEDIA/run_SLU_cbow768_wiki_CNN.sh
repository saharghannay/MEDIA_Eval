#!/usr/bin/env bash

f=100
working_dir=""
trained="trained_wiki"
data_dir=$working_dir"/Data/"$trained
embed="cbow_768"
train=$data_dir"/"$embed"/cbow_train_wmanuel_lref.u8.txt"
dev=$data_dir"/"$embed"/cbow_dev_wmanuel_lref.u8.txt"
test_=$data_dir"/"$embed"/cbow_test_wmanuel_lref.u8.txt"
res_dir=$working_dir"results/MEDIA/freeze/"$trained"/"$embed
model_name=$trained"_"$embed_"CNN_char_dim_$f"



echo $res_dir


if [ ! -d $res_dir ]; then
   mkdir -p $res_dir
   mkdir $res_dir"/models"
   mkdir $res_dir"/log"
   mkdir $res_dir"/predictions"
fi

n=3

for m in LSTM; do
  for b in {16,32,64,128}; do
    for h in {128,256,512}; do

	   python SLU_BLSTM_freeze_CNN.py --mode $m --num_epochs 200 --batch_size $b --hidden_size $h --num_layers $n\
         --char_dim $f --num_filters $f --tag_space 0 \
        --learning_rate 0.001 --decay_rate 0.05 --schedule 1 --gamma 0.0 --task MEDIA --optim ADAM \
         --dropout std --p 0.5 --unk_replace 0.0 --bidirectional True\
        --data_path $res_dir --modelname $model_name  --train $train  --dev $dev  --test $test_
done
done
done
