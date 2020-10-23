#!/usr/bin/env bash

working_dir=""
trained="trained_media"
data_dir=$working_dir"/Data/data_MEDIA"
train=$data_dir"/train_wmanuel_lref.u8.txt"
dev=$data_dir"/dev_wmanuel_lref.u8.txt"
test_=$data_dir"/test_wmanuel_lref.u8.txt"
embeddings_dir=$working_dir"/Embeddings/"$trained
embedding=$embeddings_dir"/trained_media/embed_MEDIA_data_CbowMEDIA_train_dev_data_new-iter5_vec_size_300.txt"
embed_type="cbow"
res_dir=$working_dir"/results/MEDIA/no_freeze/"$trained"/"$embed_type
model_name=$train_embed"_"$embed_type"_CNN"



if [ ! -d $res_dir ]; then
   mkdir -m $res_dir
   mkdir $res_dir"/models"
   mkdir $res_dir"/log"
   mkdir $res_dir"/predictions"
fi

for n in {1,2,3}; do
for m in LSTM; do
 for b in {16,32,64,128}; do
   for h in {128,256,512}; do

	python SLU_BLSTM_nofreeze_CNN.py  --mode $m  --num_epochs 200  --batch_size $b --hidden_size $h  --num_layers $n \
        --char_dim 30 --num_filters 30 --tag_space 0 \
        --learning_rate 0.01 --decay_rate 0.05 --schedule 1 --gamma 0.0  --optim ADAM  \
        --dropout std --p 0.5   --unk_replace 0.0  \
        --embedding glove --embedding_dict $embedding  --bidirectional True \
        --train $train --dev $dev  --test $test_  --data_path $res_dir --modelname $model_name

done 
done
done
done