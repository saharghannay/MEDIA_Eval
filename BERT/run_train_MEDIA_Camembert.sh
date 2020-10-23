export MAX_LENGTH=256
export BERT_MODEL=BERT_models/camembert-base
export data_dir=../data/MEDIA_inv
cat $data_dir"/train.txt" $data_dir"/dev.txt" $data_dir"/test.txt" | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $data_dir"/labels.txt"
export OUTPUT_DIR=FineTune_CamemBERT/MEDIA_camembert-base
export BATCH_SIZE=16
export NUM_EPOCHS=100
export SAVE_STEPS=1500
export SEED=1
python3 run_ner.py --data_dir $data_dir \
--model_type camembert \
--labels $data_dir"/labels.txt" \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--num_train_epochs 100 \
--overwrite_output_dir \
--keep_accents \
--do_train \
--do_eval \
--do_predict
