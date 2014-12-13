from yagoFeatures import yagoScores
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
from yagoFeatures import yagoScores
output_file = DictWriter(open("train_data.csv", 'w'), ['Question ID','Question Text','QANTA Scores','IR_Wiki Scores', 'Answer','Sentence Position','category','qanta t2_count','wiki t2_count','qanta t1_t3_count','wiki t1_t3_count'])
output_file.writeheader()
training_file = "/home/sivacu/Desktop/nlp/Proj/fullData/train.csv"
entries = DictReader(open(training_file, 'r'))
yago = yagoScores()


for entry in entries:
    text = entry['Question Text']
    qanta_t2 = ""
    wiki_t2 = ""
    qanta_t1_t3 = ""
    wiki_t1_t3 = ""
    for each in entry['QANTA Scores'].split(","):
        #print each
        g = each.split(":")[0]
        yScores = yago.getScore(text,g)
    	qanta_t2 +=  str(yScores[0])
    	qanta_t2 += ","
        qanta_t1_t3 +=  str(yScores[1])
        qanta_t1_t3 += ","
    for each in entry['IR_Wiki Scores'].split(","):
        g = each.split(":")[0]        	
    	yScores = yago.getScore(text,g)
        wiki_t2 +=  str(yScores[0])
    	wiki_t2 += ","
        wiki_t1_t3 +=  str(yScores[1])
        wiki_t1_t3 += ","

    output_file.writerow({
        'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'], 'Answer':entry['Answer'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'qanta t2_count':qanta_t2,'wiki t2_count':wiki_t2,'qanta t1_t3_count':qanta_t1_t3,'wiki t1_t3_count':wiki_t1_t3})

