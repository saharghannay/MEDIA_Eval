#!/bin/csh


set working_dir=""
set script="prepare_data.py"
set train=$data_dir"train_wmanuel_lref.u8.txt"
set dev=$data_dir"dev_wmanuel_lref.u8.txt"
set test=$data_dir"test_wmanuel_lref.u8.txt"

set trained_em="MEDIA"
set embeddings_dir=$working_dir"/Embeddings/trained_media/"
set cbow=$embeddings_dir"embed_MEDIA_data_CbowMEDIA_train_dev_data_new-iter5_vec_size_300.txt"
set res_dir=$working_dir"/Data/trained_"$trained_em"/cbow"




if (! -d $res_dir  ) then
    mkdir -p $res_dir
endif


foreach j ($dev $train $test)
   python  $script --input_features $cbow  --input_data $j --embeddings_type "cbow" --output $res_dir
end


