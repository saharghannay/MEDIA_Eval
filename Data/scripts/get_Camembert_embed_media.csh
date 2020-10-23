#!/bin/csh

set working_dir=""
set script="get_Camembert_embeddings.py"
set data_dir=$working_dir"/Data/data_MEDIA"
set train=$data_dir"/train_wmanuel_lref.u8.txt"
set test=$data_dir"/test_wmanuel_lref.u8.txt"
set dev=$data_dir"/dev_wmanuel_lref.u8.txt"
set res_dir=$working_dir"Embeddings/Camebert_Embed"

if ( ! -d $res_dir ) then
	mkdir -p $res_dir
endif

python $script --input_file $train --output_file $res_dir"/train_camember_base_ccnet_embed.txt"
python $script --input_file $test --output_file $res_dir"/test_camember_base_ccnet_embed.txt"
python $script --input_file $dev --output_file $res_dir"/dev_camember_base_ccnet_embed.txt"






