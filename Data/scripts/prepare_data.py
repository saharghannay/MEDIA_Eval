# -*- coding: utf-8 -*-
"""
## Sahar Ghannay
## LIMSI
## 22th june 2020
"""
import numpy as niip
import sys
import gzip
import io
import re
import codecs
import os
import argparse

parser = argparse.ArgumentParser(description='prepare data for experiments : word embeddings vector label .')
parser.add_argument("--input_features", default='', type=str, help="Camembert/glove,cbow,skipgram,fasttext,  embeddings file ")
parser.add_argument("--input_data", default='', type=str, help="input data :  word tag ")
parser.add_argument("--embeddings_type", default='glove', type=str, help=" Camembert or glove ")
parser.add_argument("--output_dir", default='', type=str, help="output file")
parser.add_argument("--vector_size", default=300, type=str, help="vector size")
args = parser.parse_args()




def load_dict(f):
    dict_em={}
    for line in f : 
        line=line.strip().split(" ")
        dict_em[line[0]]=line[1:len(line)]
    return dict_em
    
def check_tag(tag):
    
    if re.match(r"^B-", tag) != None :
        tag=re.sub("^B-","",tag)
        tag=tag+"-B"
    elif re.match(r"^I-", tag) != None :
        tag=re.sub("^I-","",tag)
        tag=tag+"-I"
    return tag
def main():

    if not args.input_features : 
        print("Could not find the input file !")
        exit()
    if not args.input_data :
        print("Could not find the input data !")
        exit()
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)


    if args.embeddings_type== 'Camembert' :
        input_feat=codecs.open(args.input_features,'r', encoding='utf8')
    else:
        input_feat=load_dict(codecs.open(args.input_features,'r', encoding='utf8'))
    print (" args.input_data  ",args.input_data)
    input_data=codecs.open(args.input_data,'r', encoding='utf8')
    output=codecs.open(args.output_dir+'/%s_%s'% (args.embeddings_type,os.path.basename(args.input_data)),'w', encoding='utf8')
    
    if args.embeddings_type== 'Camembert' :
    
        sent_id=1
        for features, data  in zip(input_feat, input_data):
            line_feat= re.sub(' ','',features.strip())
            line_data= re.sub(' ','',data.strip())
            if  data.strip()=='':
                output.write("\n")
                sent_id+=1
            else:
                line_feat= features.strip().split(" ")
                line_data= data.strip().split(" ")
                if len(line_data) ==1:
                    print ("problème ", line_data)
                mot=line_data[0]
                tag=line_data[1]
                embeddings=line_feat[1:len(line_feat)]
                output.write(str(sent_id)+" "+mot+" "+tag+" "+" ".join(embeddings)+"\n")

    else :  # glove embeddings : 
        sent_id=1
        for data  in  input_data:
            line_data= data.strip().split(" ")
            if data.strip() == '' :
                output.write("\n")
                sent_id+=1
            else:
                mot=line_data[0]
                tag=line_data[1]
                if mot in  input_feat:
                    embeddings=input_feat[mot]
                elif mot.lower() in  input_feat:
                    embeddings=input_feat[mot.lower()]
                else: 
                    houtput=[0]*args.vector_size
                    embeddings=" ".join([str(x) for x in houtput])
                output.write(str(sent_id)+" "+mot+" "+tag+" "+" ".join(embeddings)+"\n")







if __name__ == "__main__":
     main()



