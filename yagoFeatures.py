# -*- coding: utf-8 -*-
from csv import DictReader, DictWriter
import argparse
from collections import defaultdict
from score_combiner import ScoreCombiner
from operator import itemgetter
from nltk.tag.stanford import POSTagger
import re
#import wikipedia
import mysql.connector

class yagoScores:
    def __init__(self):
        self.cnx = mysql.connector.connect(user='root', database='yago', password = 'root')
        self.cursor = self.cnx.cursor(buffered=True)
        self.query = "select * from yagoFacts where t1='%s' or t3='%s'"
        self.en_postagger = POSTagger('parser/models/english-bidirectional-distsim.tagger', 'parser/stanford-postagger.jar')
    
        
    def parse(self,text):
        return self.en_postagger.tag(text.split())
        
    def get_underscoreWords(self,text):
        return re.findall("[a-z]+_[a-z]+", text)
    
    def findNounsSeq(self,tuples):
        self.noun = []    
        self.nouns = []
        prev = ""
        for each in tuples:
            if(each[1]=="NN"):
                self.noun.append(each[0])
            if(each[1]=="NNS"):
                self.nouns.append(prev+" "+each[0])
                prev = prev+" "+each[0]
            else:
                prev = each[0]
    def changeToYagoFormat(self, g):
        g=g.strip()
        char = [c for c in g]
        char[0] = char[0].upper()
        prev = False 
        for i in range(0,len(g)):
            if(prev == True):
                char[i] = char[i].upper()
                prev = False;
            if(char[i]=="_"):
                prev = True;
        return "<"+"".join(char)+">"
    
    def getFacts(self, g):
        facts = []
        print self.query%(g,g)
        self.cursor.execute(self.query%(g,g))
        for each in self.cursor:
            #each = each.replace("<","")
            #each = each.replace(">","")
            facts.append([each[1],each[2],each[3]])
        return facts
    def generateFeatures(self,tuples,facts):
        verbs = 0
        t1_t3 = 0
        for each in tuples:
            for f in facts:
                print f
                f = str(f[0].encode('ascii', errors='backslashreplace'))+str(f[1].encode('ascii', errors='backslashreplace'))+str(f[2].encode('ascii', errors='backslashreplace'))
                f.replace("\\",'');
                print f
                
                if(str(each[0]).lower() in f.lower()):
                    verbs += 1
        # TODO Now returns only total similarities
        return verbs;
    def searchInYago(self,tuples,guessess):

        for eachGuess in guessess:
            eachGuess = self.changeToYagoFormat(eachGuess)
            facts = self.getFacts(eachGuess)
            #print facts
            features = self.generateFeatures(tuples,facts)
            print features
            
            
    # Call getScore(self,text,guessess)function from outside, returns dict of scores of wiki appearances
    def getScore(self,text,guessess):
        self.freq = defaultdict(int)
        tuples = self.parse(text)
        print tuples
        self.findNounsSeq(tuples)
        self.searchInYago(tuples,guessess)
        

def main():
    text = "charles lorencez led 6000 troops in a frontal assault up the cerro de guadalupe. porfirio d \ iaz led a cavalry company against the attacker 's flank allowing general zaragoza to defeat the french."
    guessess = ['free_soil_party','equal_rights_amendment','thaddeus_stevens','barry_goldwater','woodrow_wilson','wilmot_proviso','hubert_humphrey','john_c._calhoun','grover_cleveland','warren_g._harding']
    guessess = ['battle_of_puebla','pancho_villa','battle_of_chancellorsville','francisco_i._madero',' arthur_wellesley,_1st_duke_of_wellington','battle_of_antietam','charles_martel','battle_of_austerlitz','george_b._mcclellan','battle_of_blenheim','credit_mobilier_of_america_scandal']
    ob = yagoScores()
    ob.getScore(text,guessess)
        
main()