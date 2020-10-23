#!/bin/csh

set working_dir=""
set script="prepare_data.py"
set train=$data_dir"train_wmanuel_lref.u8.txt"
set dev=$data_dir"dev_wmanuel_lref.u8.txt"
set test=$data_dir"test_wmanuel_lref.u8.txt"

set trained_em="wiki"
set embeddings_dir=$working_dir"/Embeddings/trained_wiki"
set cbow_em=$embeddings_dir"embed_MEDIA_data_Cbowwiki_fr_data_iter5_vec_size_300.txt"
set cbow_em_768=$embeddings_dir"Cbowwiki_fr_data_iter5_new_vec_size_768.txt"
set res_dir=$working_dir"/Data/trained_"$trained_em



if (! -d $res_dir  ) then
    mkdir $res_dir
endif

foreach i (cbow_768 cbow)#cbow_cor skipgram_cor  glove_cor)# ELMO) 

    if (! -d $res_dir"/"$i) then
        mkdir $res_dir"/"$i
    endif
end
foreach j ($dev $train $test)
   python  $script --input_features $cbow_em  --input_data $j --embeddings_type "cbow" --output $res_dir"/cbow"
   python  $script --input_features $cbow_em_768 --vector_size 768   --input_data $j --embeddings_type "cbow" --output_dir  $res_dir"/cbow_768"
end

