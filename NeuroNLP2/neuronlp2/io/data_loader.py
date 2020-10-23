__author__ = 'max'

import os.path
import random
import numpy as np
from .reader import CoNLL03Reader, DataReader
from .alphabet import Alphabet
from .logger import get_logger
import utils
import torch
from torch.autograd import Variable

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
PAD_POS = b"_PAD_POS"
PAD_CHUNK = b"_PAD_CHUNK"
PAD_target = b"_PAD_target"
PAD_CHAR = b"_PAD_CHAR"
_START_VOCAB = [PAD,]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0
#2 # we have too labels (0/1)

NUM_SYMBOLIC_TAGS = 1

_buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 140]


def get_feature_dim(train_path):
    with open(train_path, 'r') as file:
 	line=file.readline()
        tokens=line.strip().split(' ')
        return len(tokens[3:len(tokens)])

def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=50000, 
                     min_occurence=1, normalize_digits=True):

  
    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split(' ')
                    word = tokens[1]#utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                    target = tokens[2]

                    target_alphabet.add(target)

                    if word not in vocab_set :
                        vocab_set.add(word)
                        vocab_list.append(word)


    #logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    target_alphabet = Alphabet('target')
    feature_vect_dim = 0
    if not os.path.isdir(alphabet_directory):
        print("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)

#        target_alphabet.add(PAD_target)
        vocab = dict()
        with open(train_path, 'r') as file:
            for line in file:
                line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split(' ')
                
                for char in tokens[1]:
                    char_alphabet.add(char)

                word = tokens[1]#utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                target = tokens[2]
		feature_vect_dim = len(tokens[3:len(tokens)])
                target_alphabet.add(target)

                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count < 1])
        # if a singleton is in pretrained embedding dict, set the count to 2

        #vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_list =  sorted(vocab, key=vocab.get, reverse=True)
        print("Total Vocabulary Size: %d" % len(vocab_list))
        print("TOtal Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        print("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        target_alphabet.save(alphabet_directory)
    else:
        print ("loading alphabet")
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        target_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    target_alphabet.close()
    print("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    print("Character Alphabet Size: %d" % char_alphabet.size())
    print("target Alphabet Size: %d" % target_alphabet.size())
    return word_alphabet, char_alphabet,  target_alphabet


def read_data(source_path, word_alphabet, char_alphabet, target_alphabet, max_size=None,
              normalize_digits=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
  #  print('Reading data from %s' % source_path)
    counter = 0
    reader = DataReader(source_path, word_alphabet, char_alphabet, target_alphabet)
    inst = reader.getNext(normalize_digits)
 #   print " _buckets  ", _buckets
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)
     #   print " max size  ", max_size , "counter ", counter 
        inst_size = inst.length()
        sent = inst.sentence
        dim_feat_vect=sent.dim_feat_vec
    #    print " inst_size  ", inst_size
    #    print " sent.words ", sent.word_ids 
        for bucket_id, bucket_size in enumerate(_buckets):
   #         print " bucket_id  ", bucket_id
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, sent.feature_vectors, sent.sentences_id, inst.target_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                 
  #              print " max_char_length[bucket_id]   ", max_char_length[bucket_id] ," max_len ", max_len 
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
 #                   print " max_char_length[bucket_id]   ", max_char_length[bucket_id] 
                break

        inst = reader.getNext(normalize_digits)
    reader.close()
    print("Total number of data: %d" % counter)
#    print (" data ",data ," max_char_length ", max_char_length)
    return data, max_char_length, dim_feat_vect


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    chid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    nid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, chids, nids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
    #    print "  wid_inputs  ", wid_inputs
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # chunk ids
        chid_inputs[b, :inst_size] = chids
        chid_inputs[b, inst_size:] = PAD_ID_TAG
        # ner ids
        nid_inputs[b, :inst_size] = nids
        nid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks[b, :inst_size] = 1.0

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, chid_inputs, nid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, chids, nids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # chunk ids
            chid_inputs[i, :inst_size] = chids
            chid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], chid_inputs[excerpt], \
                  nid_inputs[excerpt], masks[excerpt]


def read_data_to_variable(source_path, word_alphabet, char_alphabet,  target_alphabet,
                          max_size=None, normalize_digits=True, use_gpu=False, volatile=False):
    data, max_char_length,dim_feat_vect = read_data(source_path, word_alphabet, char_alphabet, target_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    print " dim_feat_vect ", dim_feat_vect
    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue
        
        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
       # print " char_length  ", char_length
       # print " bucket_length  ", bucket_length
       # print " bucket_size ", bucket_size
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        sentid_inputs = np.zeros([bucket_size, bucket_length],dtype=np.int64)
        wfeat_inputs = np.zeros([bucket_size, bucket_length, dim_feat_vect], dtype=np.float32)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)

        lengths = np.empty(bucket_size, dtype=np.int64)
#        print " type  sentid_inputs  ", type(sentid_inputs)
        for i, inst in enumerate(data[bucket_id]):
      #      print " inst  ", inst
            wids, cid_seqs, fivec, sent_ids, tids = inst
     #       print " wids ",wids, " cid_seqs ",cid_seqs," pids ",pids, " chids ",chids, " nids  "  ,nids
            inst_size = len(wids)
            lengths[i] = inst_size
    #        print " inst_size ",  inst_size, " i ", i
            # sent_ids
            sentid_inputs[i, :inst_size] = sent_ids
            #feat vectors 
            wfeat_inputs[i,:inst_size] = fivec
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # ner ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            """
            I don t need to replace non frequent words to unk
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1
            """
 #       print " type wfeat_inputs ", type(wfeat_inputs)
        with torch.no_grad():
             features = Variable(torch.from_numpy(wfeat_inputs))
  #      print " type features  ", type(features)
             words = Variable(torch.from_numpy(wid_inputs))
             sents = Variable(torch.from_numpy(sentid_inputs))
             chars = Variable(torch.from_numpy(cid_inputs))
             targets = Variable(torch.from_numpy(tid_inputs))
             masks = Variable(torch.from_numpy(masks))
             lengths = torch.from_numpy(lengths)
        #we have to load batch by batch in the gpu
        if use_gpu:
            words = words.cuda()
            chars = chars.cuda()
            features = features.cuda()
            sents = sents.cuda()
            masks = masks.cuda()
            lengths = lengths.cuda()
            targets = targets.cuda()
        data_variable.append((words, features, sents,chars, targets, masks, lengths))
  #  print " len data_varibale ", len (data_variable) , " bucket_sizes   ", bucket_sizes
    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, unk_replace=0., use_gpu=False):
    data_variable, bucket_sizes = data
  #  print " ========================================================================="
   # print " data_variable " , data_variable
  #  print " bucket_sizes ", bucket_sizes
    total_size = float(sum(bucket_sizes))
  #  print  " total_size ", total_size
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

   # print " buckets_scale  ", buckets_scale
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
   # print " bucket_id  ", bucket_id
    bucket_length = _buckets[bucket_id]
   # print " bucket_length  ", bucket_length
    words, features, sents, chars, target, masks, lengths = data_variable[bucket_id]
   # print " words  ", words 
   # print "features ", features, " type ", type(features)
   # print " sents ", sents, " type ", sents
    bucket_size = bucket_sizes[bucket_id]
  #  print "min  bucket_size  ", bucket_size, " batch_size ", batch_size
    batch_size = min(bucket_size, batch_size)
   # print " batch_size  ", batch_size
    index = torch.randperm(bucket_size).long()[:batch_size]
   # print " index ", index
   # print " target ", target
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
        noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
        words = words * (ones - single[index] * noise)
    """
    if use_gpu:
       words = words.cuda()
       
       features[index] = features[index].cuda()
       chars[index] = chars[index].cuda()
       target[index] = target[index].cuda()
       masks[index] = masks[index].cuda()
       lengths[index] = lengths[index].cuda()
    """
    return words, features[index], sents[index], chars[index],  target[index], masks[index], lengths[index]


def iterate_batch_variable(data, batch_size, unk_replace=0., shuffle=False, use_gpu=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue
        words, features, sents, chars, targets, masks, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
            noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
                """
                if use_gpu:
                   words[excerpt] = words[excerpt].cuda()
                   sents[excerpt] = sents[excerpt].cuda()
                   features[excerpt] = features[excerpt].cuda()
                   chars[excerpt] = chars[excerpt].cuda()
                   target[excerpt] = target[excerpt].cuda()
                   masks[excerpt] = masks[excerpt].cuda()
                   lengths[excerpt] = lengths[excerpt].cuda()
                """
      
    #        print " len chars[index] ", len(chars[excerpt])
    #        print " features[index] ", features[excerpt]
    #        print " sents[index] ", sents[excerpt]
    #        print " chars[index] ", chars[excerpt] 
    #        print " masks[index] ", masks[excerpt]
    #        print " lengths[index] ", lengths[excerpt]
   # 	    print "target[index] ", targets[excerpt]
            yield words[excerpt], features[excerpt], sents[excerpt], chars[excerpt], targets[excerpt], masks[excerpt], lengths[excerpt]
