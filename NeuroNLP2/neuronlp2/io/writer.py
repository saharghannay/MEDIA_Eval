__author__ = 'max'

from .evaluation import *

class SLUWriter(object):
    def __init__(self, word_alphabet, char_alphabet, target_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__target_alphabet = target_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word,  predictions,  targets, lengths):
        batch_size, _ = word.shape
        #print " word ", word
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                tgt = self.__target_alphabet.get_instance(targets[i, j]).encode('utf-8')
                pred = self.__target_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                #self.__source_file.write('%d  %d %s %f %s %s \n' % (j + 1, st, w,pr, tgt, pred))
                self.__source_file.write('%d %s %s %s \n' % (j + 1,  w, tgt, pred))
            self.__source_file.write('\n')





class DataWriter(object):
    def __init__(self, word_alphabet, char_alphabet, target_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__target_alphabet = target_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write_output(self, word, sent, predictions,  targets, lengths, Houtputs):
        batch_size, _ = word.shape
        #print " word ", word
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                st = sent[i, j]
                tgt = self.__target_alphabet.get_instance(targets[i, j]).encode('utf-8')
                pred = self.__target_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                #self.__source_file.write('%d  %d %s %f %s %s \n' % (j + 1, st, w,pr, tgt, pred))
		houtput=Houtputs[i,j].numpy()
                self.__source_file.write('%d  %d %s %s %s %s\n' % (st, j + 1, w, tgt, pred," ".join([str(x) for x in houtput])))
            self.__source_file.write('\n')

    def write(self, word, sent, predictions,  targets, lengths):
        batch_size, _ = word.shape
        #print " word ", word
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                st = sent[i, j]
                tgt = self.__target_alphabet.get_instance(targets[i, j]).encode('utf-8')
                pred = self.__target_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                #self.__source_file.write('%d  %d %s %f %s %s \n' % (j + 1, st, w,pr, tgt, pred))
                self.__source_file.write('%d  %d %s %s %s \n' % (j + 1, st, w, tgt, pred))
            self.__source_file.write('\n')

    def tensor_to_list (self, predictions, targets,lengths): # tag can be ref labels or predictions
            correct_tag=[]
            pred_tag=[]
            batch_size, _ = targets.shape

            for i in range(batch_size):
                sent_c=[]
                sent_p=[]
                for j in range(lengths[i]):
                    tgt = self.__target_alphabet.get_instance(targets[i, j]).encode('utf-8')
                    pred = self.__target_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                    #if tgt != 'null' and tgt != 'O' :
                    sent_c.append(preprocess_label(tgt))
                    sent_p.append(preprocess_label(pred))
                if len(sent_c) !=0:
                    correct_tag.append(sent_c)
                    pred_tag.append(sent_p)
            return correct_tag, pred_tag

    def evaluate (self, predictions, targets):
        correct=0
        count=0
        for i in range (len(targets)):
            for j in range (len(targets[i])):
                if targets[i][j] == 'null' or targets[i][j] =='O' :
                   if predictions[i][j] != targets[i][j] :
                        count+=1
                else:
                    count+=1
                    if targets[i][j] == predictions[i][j] :
                       correct+=1
        pres, rec, f1 = compute_f1(predictions,targets)
        return pres*100, rec*100, f1*100,float(correct)*100/count


class CoNLL03Writer(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, chunk, predictions, targets, lengths):
        batch_size, _ = word.shape
        for i in range(batch_size):
            for j in range(lengths[i]):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_alphabet.get_instance(pos[i, j]).encode('utf-8')
                ch = self.__chunk_alphabet.get_instance(chunk[i, j]).encode('utf-8')
                tgt = self.__ner_alphabet.get_instance(targets[i, j]).encode('utf-8')
                pred = self.__ner_alphabet.get_instance(predictions[i, j]).encode('utf-8')
                self.__source_file.write('%d %s %s %s %s %s\n' % (j + 1, w, p, ch, tgt, pred))
            self.__source_file.write('\n')


class CoNLLXWriter(object):
    def __init__(self, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = None
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__type_alphabet = type_alphabet

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, head, type, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                w = self.__word_alphabet.get_instance(word[i, j]).encode('utf-8')
                p = self.__pos_alphabet.get_instance(pos[i, j]).encode('utf-8')
                t = self.__type_alphabet.get_instance(type[i, j]).encode('utf-8')
                h = head[i, j]
                self.__source_file.write('%d\t%s\t_\t_\t%s\t_\t%d\t%s\n' % (j, w, p, h, t))
            self.__source_file.write('\n')
