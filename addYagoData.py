from math import sqrt
from itertools import count
from operator import itemgetter
from collections import defaultdict
from csv import DictReader, DictWriter
import argparse
import time

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from score_combiner import ScoreCombiner
from yagoFeatures2 import yagoScores
output_file = DictWriter(open("train_data_yago_pos.csv", 'w'), ['Question ID','Question Text','QANTA Scores','IR_Wiki Scores','Sentence Position','category','qanta noun','wiki noun','qanta verb','wiki verb','qanta other','wiki other'])
output_file.writeheader()
training_file = "/home/sivacu/Desktop/nlp/Proj/fullData/train.csv"
#training_file = "/home/sivacu/Desktop/nlp/Proj/fullData/test.csv"
entries = DictReader(open(training_file, 'r'))
yago = yagoScores()

i = 0
for entry in entries:
    i+=1
    print i
    text = entry['Question Text']
    if(int(entry['Sentence Position']) > 2):
        continue;
    print entry['Sentence Position']
    qanta_nn = ""
    wiki_nn = ""
    qanta_vv = ""
    wiki_vv = ""
    qanta_ot = ""
    wiki_ot = ""
    
    for each in entry['QANTA Scores'].split(","):
        #print each
        g = each.split(":")[0]
        yScores = yago.getScore(text,g)
    	qanta_nn +=  str(yScores[0])
    	qanta_nn += ","
        qanta_vv +=  str(yScores[1])
        qanta_vv += ","
        qanta_ot +=  str(yScores[2])
        qanta_ot += ","

    for each in entry['IR_Wiki Scores'].split(","):
        g = each.split(":")[0]        	
    	yScores = yago.getScore(text,g)
        wiki_nn +=  str(yScores[0])
        wiki_nn += ","
        wiki_vv +=  str(yScores[1])
        wiki_vv += ","
        wiki_ot +=  str(yScores[2])
        wiki_ot += ","

    #output_file.writerow({
    #    'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'qanta t2_count':qanta_t2,'wiki t2_count':wiki_t2,'qanta t1_t3_count':qanta_t1_t3,'wiki t1_t3_count':wiki_t1_t3})
    output_file.writerow({
        'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'qanta noun':qanta_nn,'wiki noun':wiki_nn,'qanta verb':qanta_vv,'wiki verb':wiki_vv,'qanta other':qanta_ot,'wiki other':wiki_ot})

