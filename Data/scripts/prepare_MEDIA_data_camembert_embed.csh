#!/bin/csh

set working_dir=""
set script="prepare_data.py"
set data_dir=$working_dir"/Data/data_MEDIA"
set train=$data_dir"train_wmanuel_lref.u8.txt"
set dev=$data_dir"dev_wmanuel_lref.u8.txt"
set test=$data_dir"test_wmanuel_lref.u8.txt"
set trained_em="camembert"
set embeddings_dir=$working_dir"Embeddings/Camebert_Embed"
set train_em=$embeddings_dir"/train_camember_base_ccnet_embed.txt"
set dev_em=$embeddings_dir"/dev_camember_base_ccnet_embed.txt"
set test_em=$embeddings_dir"/test_camember_base_ccnet_embed.txt"
set res_dir=$working_dir"/Data/camembert/camembert/base_ccnet"


if (! -d $res_dir  ) then
    mkdir -p $res_dir
endif



# camembert embeddings Last hidden BLSTM layer
python  $script --input_features $train_em  --input_data $train  --embeddings_type "Camembert" --output $res_dir
python  $script --input_features $dev_em  --input_data $dev  --embeddings_type "Camembert" --output $res_dir
python  $script --input_features $test_em  --input_data $test  --embeddings_type "Camembert" --output $res_dir



