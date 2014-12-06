__author__ = 'Patrick'

import argparse
from numpy import mean

from nltk import FreqDist

from csv import DictReader, DictWriter

argparser = argparse.ArgumentParser("Sort answers and display counts.")
argparser.add_argument("filename")
args = argparser.parse_args()

answers = FreqDist([entry['Answer'] for entry in DictReader(open(args.filename, 'r'))])

for answer in answers.items():
    print answer

print '\nAverage number occurrences of each answer:', mean(answers.values())
