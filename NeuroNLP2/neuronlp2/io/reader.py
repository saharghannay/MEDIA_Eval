__author__ = 'max'

from instance import DependencyInstance, SLUInstance, NERInstance, SentInstance 
from instance import Sentence2, Sentence
import conllx_data
import utils
import numpy as np 

class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        types = []
        type_ids = []
        heads = []

        if symbolic_root:
            words.append(conllx_data.ROOT)
            word_ids.append(self.__word_alphabet.get_index(conllx_data.ROOT))
            char_seqs.append([conllx_data.ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(conllx_data.ROOT_CHAR), ])
            postags.append(conllx_data.ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(conllx_data.ROOT_POS))
            types.append(conllx_data.ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(conllx_data.ROOT_TYPE))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[4]
            head = int(tokens[6])
            type = tokens[7]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            heads.append(head)

        if symbolic_end:
            words.append(conllx_data.END)
            word_ids.append(self.__word_alphabet.get_index(conllx_data.END))
            char_seqs.append([conllx_data.END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(conllx_data.END_CHAR), ])
            postags.append(conllx_data.END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(conllx_data.END_POS))
            types.append(conllx_data.END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(conllx_data.END_TYPE))
            heads.append(0)

        return DependencyInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, heads, types,
                                  type_ids)


class DataReader(object):
    # data format : sentence_id, word, target (label) feature vector
    def __init__(self, file_path, word_alphabet, char_alphabet,target_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__target_alphabet = target_alphabet
    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        sentences_id = []
        words = []
        word_ids = []
        feature_vectors =[]
        char_seqs = []
        char_id_seqs = []
        target_tags = []
        target_ids = []
        dim_feat_vec=0
        for tokens in lines:
            sentences_id.append(tokens[0])
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = tokens[1]	# utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
            target = tokens[2]
            #print (word, sentences_id)
            feature_vectors.append(tokens[3:len(tokens)])
            dim_feat_vec=len(tokens[3:len(tokens)])
            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))
            target_tags.append(target)
            target_ids.append(self.__target_alphabet.get_index(target))
        return SentInstance(Sentence2(sentences_id, words, word_ids, np.array(feature_vectors,dtype=np.float32),dim_feat_vec, char_seqs, char_id_seqs),
                           target_tags, target_ids)


class SLUReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, slu_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__slu_alphabet = slu_alphabet
      
    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        slutags = []
        slu_ids = []
       

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[0]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = tokens[0]#utils.DIGIT_RE.sub(b"0", tokens[0]) if normalize_digits else tokens[0]
            slu = tokens[1]
           

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            slutags.append(slu)
            slu_ids.append(self.__slu_alphabet.get_index(slu))

           

        return SLUInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), slutags, slu_ids)
                                                                                             


class CoNLL03Reader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet
    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        lines = []
        line = self.__source_file.readline()
        while line is not None and len(line.strip()) > 0:
            line = line.strip()
            line = line.decode('utf-8')
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        chunk_tags = []
        chunk_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[0]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > utils.MAX_CHAR_LENGTH:
                chars = chars[:utils.MAX_CHAR_LENGTH]
                char_ids = char_ids[:utils.MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = utils.DIGIT_RE.sub(b"0", tokens[0]) if normalize_digits else tokens[0]
            pos = tokens[1]
            chunk = tokens[2]
            ner = tokens[3]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            chunk_tags.append(chunk)
            chunk_ids.append(self.__chunk_alphabet.get_index(chunk))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs), postags, pos_ids, chunk_tags, chunk_ids,
                           ner_tags, ner_ids)
