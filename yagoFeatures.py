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
        self.query1 = "select * from yagoFacts where t1 IN %s or t3 IN %s"
        
        self.en_postagger = POSTagger('parser/models/english-bidirectional-distsim.tagger', 'parser/stanford-postagger.jar')
        self.stopwords = nltk.corpus.stopwords.words('english')    
        self.prevText = ""
        self.tuples = ""
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
    
    def getFactsLevel2(self, tup):
        facts = []
        #print self.query%(g,g)
        self.cursor.execute(self.query1%(tup,tup))
        for each in self.cursor:
            #print each
            #each = each.replace("<","")
            #each = each.replace(">","")
            facts.append([each[1],each[2],each[3]])
        return facts
    
    def getFacts(self, g):
        facts = []
        #print self.query%(g,g)
        self.cursor.execute(self.query%(g,g))
        for each in self.cursor:
            #each = each.replace("<","")
            #each = each.replace(">","")
            facts.append([each[1],each[2],each[3]])
        return facts
    def generateFeatures(self, tuples, facts):
        nn = 0
        vv = 0
        oth = 0
        cd = 0
        twolevel = set()
        print ("Next twolevel")
        for f in facts:
            twolevel.add(f[0])
            twolevel.add(f[2])
        #print (twolevel)
        #twolevel.remove(guess)
        print ("Next get twolevel facts") 
        print twolevel
        facts = self.getFactsLevel2(tuple(twolevel))  
        print ("Next got Next twolevel facts")
        for f in facts:
            #print f
            #f = str(f[0].decode('ascii', 'ignore'))+str(f[1].decode('ascii', 'ignore'))+str(f[2].decode('ascii', 'ignore'))
            #f = f.lower()
            #f = f.split(">")
            #f = f[:3]
            #print f
            for i in range(0,3):
                f[i] = str(f[0].decode('ascii', 'ignore'))
                f[i] = f[i].replace("_"," ")
                f[i] = f[i].replace("<","")
                f[i] = f[i].replace(">","")
                f[i] = f[i].lower()
                f[i] = f[i].replace(")","")
                f[i] = f[i].replace("(","")
                f[i] = f[i].replace("-"," ")                                           
            #print f
            #print tuples.split()
            
            for each in tuples:
                if(each[1] in ["NN","NNS"]):
                    e = str(each[0]).lower() 
                    e = e.replace("_"," ")
                    if(e in f[0].split() or e in f[2].split()):
                        nn += 1
                if(each[1] == "VB"):
                    e = str(each[0]).lower() 
                    e = e.replace("_"," ")
                    if(e in f[1]):
                        vv += 1
                if(each[1] == "CD"):
                    e = str(each[0]).lower() 
                    e = e.replace("_"," ")
                    if(e in f[1]):
                         cd+= 1                                                
        # TODO Now returns only total similarities
        #print verbs
        print ("returning")
        return [nn,vv,cd];
    def searchInYago(self,tuples,guess):
        eachGuess = self.changeToYagoFormat(guess)
        print ("Next get facts")
        facts = self.getFacts(eachGuess)
        #print facts
        print ("Next generateFeatures")
        count = self.generateFeatures(tuples,facts)
        return count
    # Call getScore(self,text,guessess)function from outside, returns dict of scores of wiki appearances
    # input, (text, list of guesses)
    def getScore(self,text,guessess):
        #print (text+guessess)
        self.freq = defaultdict(int)
        #print ("IN GUESSS")
        if(self.prevText != text):
            self.tuples = self.parse(text)
            self.prevText = text
            print ("PARSE DONE")
        #print tuples
        #print tuples
        #self.findNounsSeq(tuples)
        #return self.searchMultipleInYago(tuples,guessess)
        return self.searchInYago(self.tuples,guessess.strip())
        
def main():
    text = "charles lorencez led 6000 troops in a frontal assault up the cerro de guadalupe. porfirio d \ iaz led a cavalry company against the attacker 's flank allowing general zaragoza to defeat the french."
    guessess = ['free_soil_party','equal_rights_amendment','thaddeus_stevens','barry_goldwater','woodrow_wilson','wilmot_proviso','hubert_humphrey','john_c._calhoun','grover_cleveland','warren_g._harding']
    guessess = ['battle_of_puebla','pancho_villa','battle_of_chancellorsville','francisco_i._madero',' arthur_wellesley,_1st_duke_of_wellington','battle_of_antietam','charles_martel','battle_of_austerlitz','george_b._mcclellan','battle_of_blenheim','credit_mobilier_of_america_scandal']
    guessess = 'battle_of_puebla'
    ob = yagoScores()
    ob.getScore(text,guessess)
        
#main()