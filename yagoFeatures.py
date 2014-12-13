# -*- coding: utf-8 -*-
from csv import DictReader, DictWriter
import argparse
from collections import defaultdict
from score_combiner import ScoreCombiner
from operator import itemgetter
from nltk.tag.stanford import POSTagger
import re
import nltk
#import wikipedia
#import mysql.connector
import pymysql

class yagoScores:
    def __init__(self):
        self.cnx = pymysql.connect(user='root', database='yago', password = 'root')
        self.cursor = self.cnx.cursor()
        self.query = "select * from yagoFacts where t1='%s' or t3='%s'"
        self.en_postagger = POSTagger('parser/models/english-bidirectional-distsim.tagger', 'parser/stanford-postagger.jar')
        self.stopwords = nltk.corpus.stopwords.words('english')    

        
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
        g=g.replace("'","")
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
        #print self.query%(g,g)
        self.cursor.execute(self.query%(g,g))
        for each in self.cursor:
            #each = each.replace("<","")
            #each = each.replace(">","")
            facts.append([each[1],each[2],each[3]])
        return facts
    def generateFeatures(self,tuples,facts):
        t2 = 0
        t1_t3 = 0
        for f in facts:
            #print f
            f = str(f[0].decode('ascii', 'ignore'))+str(f[1].decode('ascii', 'ignore'))+str(f[2].decode('ascii', 'ignore'))
            f = f.lower()
            f = f.split(">")
            f = f[:3]
            for i in range(0,3):
                f[i] = f[i].replace("_"," ")
                f[i] = f[i].replace("<","")
                f[i] = f[i].replace(")","")
                f[i] = f[i].replace("(","")
                f[i] = f[i].replace("-"," ")                                           
            #print f
            #print tuples.split()
            for each in tuples.split():
                if(each not in self.stopwords):
                    each = str(each).lower() 
                    each = each.replace("_"," ")
                    if(len(each)>2 and each in f[1]):
                        print (each+"---"+f[1])
                        t2 += 1
                    #print each
                    #print (f[0].split())
                    if(len(each)>2 and (each in f[0].split() or each in f[2].split())):
                        print (each+"------------"+f[0]+"----"+f[2])
                        
                        t1_t3 += 1    
        # TODO Now returns only total similarities
        #print verbs
        """
        bucket = 0
        if(verbs == 0):
            bucket = 0
        elif(verbs > 1 and verbs < 6):
            bucket = 1
        elif(verbs >= 6 and verbs < 11):
            bucket = 2
        elif(verbs >= 11 and verbs < 16):
            bucket = 3
        elif(verbs >= 16 and verbs < 21):
            bucket = 4
        elif(verbs >= 21 and verbs < 26):
            bucket = 5
        elif(verbs >= 26 and verbs < 31):
            bucket = 6
        elif(verbs >= 31 and verbs < 40):
            bucket = 7
        else:
            bucket = 8    
        return bucket;
        """
        return [t2, t1_t3];
    def searchInYago(self,text,guess):
        eachGuess = self.changeToYagoFormat(guess)
        facts = self.getFacts(eachGuess)
        #print facts
        count = self.generateFeatures(text,facts)
        return count
    # Call getScore(self,text,guessess)function from outside, returns dict of scores of wiki appearances
    # input, (text, list of guesses)
    def getScore(self,text,guessess):
        #print (text+guessess)
        self.freq = defaultdict(int)
        #print ("IN GUESSS")
        #tuples = self.parse(text)
        #print tuples
        #self.findNounsSeq(tuples)
        #return self.searchMultipleInYago(tuples,guessess)
        return self.searchInYago(text,guessess.strip())
        
def main():
    text = "charles lorencez led 6000 troops in a frontal assault up the cerro de guadalupe. porfirio d \ iaz led a cavalry company against the attacker 's flank allowing general zaragoza to defeat the french."
    guessess = ['free_soil_party','equal_rights_amendment','thaddeus_stevens','barry_goldwater','woodrow_wilson','wilmot_proviso','hubert_humphrey','john_c._calhoun','grover_cleveland','warren_g._harding']
    guessess = ['battle_of_puebla','pancho_villa','battle_of_chancellorsville','francisco_i._madero',' arthur_wellesley,_1st_duke_of_wellington','battle_of_antietam','charles_martel','battle_of_austerlitz','george_b._mcclellan','battle_of_blenheim','credit_mobilier_of_america_scandal']
    ob = yagoScores()
    ob.getScore(text,guessess)
        
#main()