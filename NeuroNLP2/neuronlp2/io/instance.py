__author__ = 'max'




class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)




class Sentence2(object):
    def __init__(self, sentences_id, words, word_ids, feature_vectors,dim_feat_vec, char_seqs, char_id_seqs):
        self.sentences_id = sentences_id
        self.words = words
        self.word_ids = word_ids
        self.feature_vectors = feature_vectors
        self.dim_feat_vec = dim_feat_vec
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, types, type_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.types = types
        self.type_ids = type_ids

    def length(self):
        return self.sentence.length()


class SentInstance(object):
    def __init__(self, sentence,   target_tags, target_ids):
        self.sentence = sentence
        self.target_tags = target_tags
        self.target_ids = target_ids

    def length(self):
        return self.sentence.length()
class SLUInstance(object):
    def __init__(self, sentence, slutags, slu_ids):
        self.sentence = sentence
        self.slutags = slutags
        self.slu_ids = slu_ids

    def length(self):
        return self.sentence.length()
                         

class NERInstance(object):
    def __init__(self, sentence, postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()
