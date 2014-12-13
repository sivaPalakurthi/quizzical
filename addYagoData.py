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
from wikiscores import wikiscores
output_file = DictWriter(open("train-searchWiki-Links.csv", 'w'), ['Question ID','Question Text','QANTA Scores','IR_Wiki Scores','Sentence Position','category','Answer','QANTA Wiki','Wiki Wiki','One Link'])
output_file.writeheader()
training_file = "/home/sivacu/Desktop/nlp/Proj/fullData/train.csv"
#training_file = "/home/sivacu/Desktop/nlp/Proj/fullData/test.csv"
entries = DictReader(open(training_file, 'r'))
yago = wikiscores()

i = 0
for entry in entries:
    i+=1
    qanta_nn = ""
    wiki_nn = ""
    qanta_vv = ""
    wiki_vv = ""
    qanta_ot = ""
    wiki_ot = ""
    
    print i
    text = entry['Question Text']
    if(int(entry['Sentence Position']) > 2):
        output_file.writerow({
        'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Answer':entry['Answer'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'QANTA Wiki':qanta_nn,'Wiki Wiki':wiki_nn,'One Link':''})
        continue;
    #print entry['Sentence Position']
    yScores = yago.getScore(text,"s")
    output_file.writerow({
        'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Answer':entry['Answer'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'QANTA Wiki':qanta_nn,'Wiki Wiki':wiki_nn, 'One Link':yScores})

    """
    
    
    
    one = ""
    for each in entry['QANTA Scores'].split(","):
        #print each
        g = each.split(":")[0]
        yScores = yago.getScore(text,g)
    	if(yScores != True and yScores != False):
            one = yScores

    	if(yScores==True):
       	    qanta_nn += g + ":"+ str(yScores)
       	    qanta_nn += ","
            #if(g == entry["Answer"]):
            #    print "Qanta Answer"
            #else:
            #    print "Qanta wrong"
            #
    for each in entry['IR_Wiki Scores'].split(","):
        one = ""
        g = each.split(":")[0]        	
    	yScores = yago.getScore(text,g)
        if(yScores != True and yScores != False):
            one = yScores

    	if(yScores==True):
            wiki_nn += g+":"+ str(yScores)
            wiki_nn += ","
            #if(g == entry["Answer"]):
            #    print "wiki Answer"
            #else:
            #   print "wiki wrong"
            #
    print one
    #output_file.writerow({
    #    'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'qanta t2_count':qanta_t2,'wiki t2_count':wiki_t2,'qanta t1_t3_count':qanta_t1_t3,'wiki t1_t3_count':wiki_t1_t3})
    output_file.writerow({
        'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'QANTA Wiki':qanta_nn,'Wiki Wiki':wiki_nn, 'One Link':one})

    """