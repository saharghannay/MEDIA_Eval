#!/usr/bin/python3
"""
## Sahar Ghannay
## LIMSI
## 22th june 2020
"""

import os
import argparse
import io
import re
import numpy
import torch
from transformers import CamembertModel, CamembertTokenizer


parser = argparse.ArgumentParser(description='Extract camembert embeddings .')
parser.add_argument("--input_file", default='', type=str, help="input file ")
parser.add_argument("--output_file", default='', type=str, help="output files")
args = parser.parse_args()


def readfile(f):
	input=open(f,'r')
	res=[]
	sent=[]
	for line in input:
		line=line.rstrip()
		if line != "":
			line=line.split(' ')
			sent.append(line[0])
		else:
			res.append(" ".join(sent))
			sent=[]
	return res

def main():
    if args.input_file :
        input_coprus=readfile(args.input_file)
    else :
        print("Could not find the input file !")
        exit()

    if args.output_file :
        output_file=open(args.output_file,"w")
    else :
        print("Could not find the output file !")
        exit()

    tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-base-ccnet')
    # load model
    model = CamembertModel.from_pretrained('camembert/camembert-base-ccnet')

    # read_data

    for line in input_coprus:
        line=line.rstrip()
        sent=line.split(' ')
        # encode() automatically adds the classification token <s>
        token_ids = tokenizer.encode(line)
        tokens = [tokenizer._convert_id_to_token(idx) for idx in token_ids]
        # unsqueeze token_ids because batch_size=1
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        # forward method returns a tuple (we only want the logits : last output layer)
        # squeeze() because batch_size=1
        output = model(token_ids)[0].squeeze()
        #output[0] is th output of CLS token (<s>), which is the first token, can be considered as the sentence embeddings
        # mapp each word to its sentence embeddings taking into account splitted words
        embeddings=len(sent)*[[]]
        words=len(sent)*[""]
        l=-1
        for i in range (1,len(output)-1):
            if re.match("^▁",tokens[i]) :
                # write to file 
                if l != -1:
                    if len( embeddings[l] ) >1:
                        em=numpy.array(embeddings[l])
                        word_vec=list(numpy.sum(em,axis=0))
                    else:
                        word_vec=embeddings[l][0]
                    output_file.write(words[l]+" "+" ".join([str(x) for x in word_vec])+"\n")
                
                l=l+1
                print (i, l, tokens[i])
                words[l]=re.sub("▁","",tokens[i])
                embeddings[l]=[output[i].detach().numpy()]
            else:
                print (i, l, tokens[i])
                words[l]=words[l]+tokens[i]
                print (words[l])
                embeddings[l].append(output[i].detach().numpy())

        if len( embeddings[l] ) >1:
            em=numpy.array(embeddings[l])
            word_vec=list(numpy.sum(em,axis=0))
        else:
            word_vec=embeddings[l][0]
        output_file.write(words[l]+" "+" ".join([str(x) for x in word_vec])+"\n")

        output_file.write("\n")
            





if __name__ == "__main__":
     main()

