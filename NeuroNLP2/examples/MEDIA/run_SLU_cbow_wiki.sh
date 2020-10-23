#!/usr/bin/env bash


working_dir=""
trained="trained_wiki"
data_dir=$working_dir"/Data/"$trained
embed="cbow"
train=$data_dir"/"$embed"/cbow_train_wmanuel_lref.u8.txt"
dev=$data_dir"/"$embed"/cbow_dev_wmanuel_lref.u8.txt"
test_=$data_dir"/"$embed"/cbow_test_wmanuel_lref.u8.txt"
res_dir=$working_dir"/results/MEDIA/freeze/"$trained"/"$embed
echo $res_dir


if [ ! -d $res_dir ]; then
   mkdir -p $res_dir
   mkdir $res_dir"/models"
   mkdir $res_dir"/log"
   mkdir $res_dir"/predictions"
fi

for n in {1,2,3}; do
for m in LSTM; do
  for b in {16,32,64,128}; do
    for h in {128,256,512}; do
	for f in {30,50,100}; do
		model_name=$trained"_"$embed_"CNN_char_dim_"$f

	 python SLU_BLSTM_freeze.py --mode $m --num_epochs 200 --batch_size $b --hidden_size $h --num_layers 1\
	 --char_dim 30 --num_filters 30 --tag_space 0 \
 	--learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0.0 --task MEDIA --optim SGD\
	 --dropout std --p 0.5 --unk_replace 0.0 --bidirectional True\
 	--data_path $res_dir --modelname $model_name  --train $train  --dev $dev  --test $test_
done
done
done
done
done
