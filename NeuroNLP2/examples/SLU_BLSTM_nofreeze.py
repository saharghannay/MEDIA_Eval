from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM  model for SLU 
"""

import os
import sys
import subprocess

sys.path.append(".")
sys.path.append("..")
from datetime import datetime
import time
import argparse
import uuid
import random
import numpy as np
import torch
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer, SLUWriter,  DataWriter, slu_data
from neuronlp2.models import BiRecurrentConv2, BiRecurrentConv, BiVarRecurrentConv
from neuronlp2 import utils

SEED=2

#random.seed(SEED)
#np.random.seed(SEED)
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

 
uid = uuid.uuid4().get_hex()[:6]


def evaluate(output_file, data_path, data, task, optim):
    score_file = "%s/predictions/score_%s_%s.txt" % (data_path,data,optim)
    if task == 'MEDIA' :
	os.system("eval/conlleval_inv.pl < %s > %s" % (output_file, score_file))
    else:
	os.system("eval/conlleval_inv.pl < %s > %s" % (output_file, score_file))
    
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', required=True)
    parser.add_argument('--p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words',required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--data_path') 
    parser.add_argument('--modelname', default="ASR_ERR_LSTM.json.pth.tar", help='model name') 
    parser.add_argument('--task', default="MEDIA", help='task name : MEDIA or ATIS') 
    parser.add_argument('--optim', default="SGD", help=' Optimizer : SGD or ADAM') 
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    tim=datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file='%s/log/log_model_%s_mode_%s_num_epochs_%d_batch_size_%d_hidden_size_%d_num_layers_%d_optim_%s_lr_%f_tag_space_%s.txt'%(args.data_path,args.modelname,args.mode,args.num_epochs,args.batch_size,args.hidden_size,args.num_layers, args.optim,args.learning_rate,str(args.tag_space))   
    logger = get_logger("SLU_BLSTM", log_file)
    
    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    data_path=args.data_path
    bidirectional=args.bidirectional
    p = args.p
    unk_replace = args.unk_replace
    embedding = args.embedding
    embedding_path = args.embedding_dict
    out_path=args.data_path
    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet,  target_alphabet = slu_data.create_alphabets('%s/data_dic'%(data_path), train_path,
                                                                 data_paths=[dev_path, test_path],embedd_dict=embedd_dict,
                                                                 max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("Target Alphabet Size: %d" % target_alphabet.size())
    logger.info("Bidirectionnal %s" % bidirectional)


    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = slu_data.read_data_to_variable(train_path, word_alphabet, char_alphabet,target_alphabet, use_gpu=use_gpu)
    num_data = sum(data_train[1])
    num_labels = target_alphabet.size()
    print(" num_labels", num_labels )
    data_dev = slu_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, target_alphabet, use_gpu=use_gpu, volatile=True)
    data_test = slu_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, target_alphabet, use_gpu=use_gpu, volatile=True)
    writer = SLUWriter(word_alphabet, char_alphabet,  target_alphabet)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[slu_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()

    char_dim = args.char_dim
    window = 3
    num_layers = args.num_layers
    tag_space = args.tag_space
    print(" embedd_dim ", embedd_dim)
    if args.dropout == 'std':
        network = BiRecurrentConv2(embedd_dim, word_alphabet.size(),char_dim, char_alphabet.size(),
                                  num_filters, window,
                                  mode, hidden_size, num_layers, num_labels,tag_space=tag_space,
                                  embedd_word=word_table, p_rnn=p,bidirectional=bidirectional)
    else:
        network = BiVarRecurrentConv(embedd_dim, word_alphabet.size(),
                                     char_dim, char_alphabet.size(),
                                     num_filters, window,
                                     mode, hidden_size, num_layers, num_labels,
                                     tag_space=tag_space, embedd_word=word_table, p_rnn=p)
    print (network)
    if use_gpu:
        network.cuda()

    lr = learning_rate
    if args.optim == "SGD":
    	optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    else:
    	optim = Adam(network.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=gamma)
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d" % (
        mode, num_layers, hidden_size, num_filters, tag_space))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, dropout: %.2f, unk replace: %.2f)" % (
        gamma, num_data, batch_size, p, unk_replace))
    num_batches = num_data / batch_size + 1
    dev_f1 = 0.0
    dev_acc = 0.0
    dev_precision = 0.0
    dev_recall = 0.0
    test_f1 = 0.0
    test_acc = 0.0
    test_precision = 0.0
    test_recall = 0.0
    best_epoch = 0
    model_path=""
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, args.dropout, lr, decay_rate, schedule))
        train_err = 0.
        train_corr = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
        #for batch_train in slu_data.iterate_batch_variable(data_train, batch_size):
            word, char, labels, masks, lengths = slu_data.get_batch_variable(data_train, batch_size,
                                                                                       unk_replace=unk_replace)
            optim.zero_grad()
            loss, corr, _  = network.loss(word, char, labels, mask=masks, length=lengths, leading_symbolic=slu_data.NUM_SYMBOLIC_TAGS)
            loss.backward()
            optim.step()
	   
            num_tokens = masks.data.sum()
            #train_err += loss.data * num_tokens
            train_err += loss.data[0] * num_tokens
            #train_corr += corr.data
            train_corr += corr.data[0]
            train_total += num_tokens
            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                    batch, num_batches, train_err / train_total, train_corr * 100 / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)
            batch=batch+1
        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))
        logger.info('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))
        print('train: %d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            num_batches, train_err / train_total, train_corr * 100 / train_total, time.time() - start_time))
	loss_results=train_err / train_total
        # evaluate performance on dev data
        network.eval()
	tmp_filename = '%s/predictions/dev_%s_num_layers_%s_%s.txt' % (out_path,args.optim, str(args.num_layers),str(uid))
        writer.start(tmp_filename)
        all_target=[]
        all_preds=[]
        for batch in slu_data.iterate_batch_variable(data_dev, batch_size):
            word, char,  labels, masks, lengths = batch
            _, _, preds = network.loss(word, char, labels, mask=masks, length=lengths,
                                       leading_symbolic=slu_data.NUM_SYMBOLIC_TAGS)
            writer.write(word.data.cpu().numpy(),  preds.data.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
        #    correct_tag, pred_tag=writer.tensor_to_list(preds.cpu().numpy(),labels.cpu().numpy(), lengths.cpu().numpy())
         #   all_target.extend(correct_tag)
          #  all_preds.extend(pred_tag)
        writer.close()
       # precision, recall,f1,acc=writer.evaluate(all_preds,all_target)
        acc, precision, recall, f1 = evaluate(tmp_filename, data_path,"dev", args.task,args.optim)
	print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
        logger.info('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
        if dev_acc < acc:
            dev_f1 = f1
            dev_acc = acc
            dev_precision = precision
            dev_recall = recall
            best_epoch = epoch

	    # save best model
            model_path = "%s/models/best_model_%s_mode_%s_num_epochs_%d_batch_size_%d_hidden_size_%d_num_layers_%d_bestdevacc_%f_bestepoch_%d_optim_%s_lr_%f_tag_space_%s"%(args.data_path,args.modelname,mode,num_epochs,batch_size,hidden_size,args.num_layers,dev_acc,best_epoch,args.optim,args.learning_rate,str(tag_space))        
            torch.save(network,model_path)

            # evaluate on test data when better performance detected
	    """
            tmp_filename = '%s/tmp/%s_test%d' % (data_path,tim, epoch)
            writer.start(tmp_filename)

            for batch in slu_data.iterate_batch_variable(data_test, batch_size):
                word, features, sents, char, labels, masks, lengths = batch
                _, _, preds,probs = network.loss(features, char, labels, mask=masks, length=lengths,
                                              leading_symbolic=slu_data.NUM_SYMBOLIC_TAGS)
                writer.write(word.data.cpu().numpy(),sents.data.cpu().numpy(),
                             preds.data.cpu().numpy(), probs.data.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            writer.close()
            test_acc, test_precision, test_recall, test_f1 = evaluate(tmp_filename, data_path,"test",tim)
            """

        logger.info("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
            dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
#        logger.info("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
 #           test_acc, test_precision, test_recall, test_f1, best_epoch))


        if epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
	    if args.optim == "SGD":
       	 	optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    	    else:
        	optim = Adam(network.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=gamma)



    # end epoch
    # test evaluation 
    # load model
    print ("model path ", model_path) 
    network = torch.load(model_path)
    if use_gpu :
        network.cuda()
    # mode eval
    network.eval()
    # evaluate on test dev when better performance detected
    tmp_filename = '%s/predictions/dev_best_model_%s_mode_%s_num_epochs_%d_batch_size_%d_hidden_size_%d_num_layers_%d_bestdevacc_%f_bestF1_%f_bestepoch_%d_optim_%s_lr_%f_tag_space_%s' % (out_path,args.modelname,mode,num_epochs,batch_size,hidden_size,num_layers,dev_acc,dev_f1,best_epoch,args.optim,args.learning_rate,tag_space)

    #tmp_filename = '%s/predictions/dev_best_model_%s_mode_%s_num_epochs_%d_batch_size_%d_hidden_size_%d_num_layers_%d_bestdevacc_%f_bestepoch_%d_optim_%s_lr_%f_tag_space_%s' % (out_path,args.modelname,mode,num_epochs,batch_size,hidden_size,num_layers,dev_acc,best_epoch,args.optim,args.learning_rate,tag_space)    #tmp_filename = '%s/predictions/dev_bestmodel_devacc_%f_epoch_%d' % (out_path,dev_acc, best_epoch)
    writer.start(tmp_filename)
    all_target=[]
    all_preds=[]
    for batch in slu_data.iterate_batch_variable(data_dev, batch_size):
        word,  char, labels, masks, lengths = batch
	_, _, preds = network.loss(word, char, labels, mask=masks, length=lengths,
                                              leading_symbolic=slu_data.NUM_SYMBOLIC_TAGS)
        writer.write(word.data.cpu().numpy(),  preds.data.cpu().numpy(),  labels.data.cpu().numpy(), lengths.cpu().numpy())
    writer.close()
    dev_acc, dev_precision, dev_recall,dev_f1=evaluate(tmp_filename, data_path,"dev", args.task,args.optim)


    # evaluate on test data when better performance detected
    tmp_filename = '%s/predictions/test_best_model_%s_mode_%s_num_epochs_%d_batch_size_%d_hidden_size_%d_num_layers_%d_bestdevacc_%f_bestF1_%f_bestepoch_%d_optim_%s_lr_%f_tag_space_%s' % (out_path,args.modelname,mode,num_epochs,batch_size,hidden_size,num_layers,dev_acc,dev_f1,best_epoch,args.optim,args.learning_rate,tag_space)
#    tmp_filename = '%s/predictions/test_best_model_%s_mode_%s_num_epochs_%d_batch_size_%d_hidden_size_%d_num_layers_%d_bestdevacc_%f_bestepoch_%d_optim_%s_lr_%f_tag_space_%s' % (out_path,args.modelname,mode,num_epochs,batch_size,hidden_size,num_layers,dev_acc,best_epoch, args.optim, args.learning_rate, tag_space)
    writer.start(tmp_filename)
    all_target=[]
    all_preds=[]
    for batch in slu_data.iterate_batch_variable(data_test, batch_size):
        word, char, labels, masks, lengths = batch
        _, _, preds = network.loss(word, char, labels, mask=masks, length=lengths,
                                              leading_symbolic=slu_data.NUM_SYMBOLIC_TAGS)
        writer.write(word.data.cpu().numpy(),
                             preds.data.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
    writer.close()
    test_acc, test_precision, test_recall,test_f1=evaluate(tmp_filename, data_path,"test", args.task,args.optim)
    print("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
    print("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (test_acc, test_precision, test_recall, test_f1, best_epoch))
    logger.info("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
    logger.info("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (test_acc, test_precision, test_recall, test_f1, best_epoch))

if __name__ == '__main__':
    main()
