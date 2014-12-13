# -*- coding: utf-8 -*-
from csv import DictReader, DictWriter
import argparse
from collections import defaultdict
from score_combiner import ScoreCombiner
from operator import itemgetter
from nltk.tag.stanford import POSTagger
import re
import wikipedia

class wikiscores:
    def __init__(self):
        None
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
    
    def searchInWiki(self,text,guessess):
        #text = " ".join(self.noun)+" ".join(self.nouns)  
        #text = " ".join(self.nouns) 
        #print text  
        links = wikipedia.search(text)
        #print ("LINKS")
        #print links    
        for link in links:
            link = link.lower()
            link = link.replace(" ","_")
            #page = wikipedia.page(link)
            #print page.title
            #check if guess appears in that page
            #print (guessess +"-----"+ link)
            if(guessess == link):
                print ("FOUNDDDDDDDDDDd")
                return True
            """for eachg in guessess:
                print eachg.replace("_", " ").lower()
                if(eachg.replace("_", " ").lower() in page.content.lower()):
                    print "founddddddddddddddddddddd"
                    self.freq[eachg] += 1
            """
        return False
    # Call getScore(self,text,guessess)function from outside, returns dict of scores of wiki appearances
    def getScore(self,text,guessess):
        #self.freq = defaultdict(int)
        #tuples = self.parse(text)
        #print tuples
        #self.findNounsSeq(tuples)
        return self.searchInWiki(text,guessess)
        #print self.freq
        #return self.freq

def main():
    text = "charles lorencez led 6000 troops in a frontal assault up the cerro de guadalupe. porfirio d \ iaz led a cavalry company against the attacker 's flank allowing general zaragoza to defeat the french."
    guessess = ['free_soil_party','equal_rights_amendment','thaddeus_stevens','barry_goldwater','woodrow_wilson','wilmot_proviso','hubert_humphrey','john_c._calhoun','grover_cleveland','warren_g._harding']
    guessess = ['battle_of_puebla','pancho_villa','battle_of_chancellorsville:','francisco_i._madero:',' arthur_wellesley,_1st_duke_of_wellington','battle_of_antietam','charles_martel','battle_of_austerlitz','george_b._mcclellan','battle_of_blenheim','credit_mobilier_of_america_scandal']
    ob = wikiscores()
    ob.getScore(text,guessess)
        
#main()