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
from wikiscores import wikiscores
#import matplotlib.pyplot as plt

kTOKENIZER = TreebankWordTokenizer()

# DEBUG
counter = count()

def morphy_stem(word):
    stem = wn.morphy(word)
    return stem.lower() if stem else word.lower()

def tokenize_and_stem(text):
    return [morphy_stem(word.strip('.')) for word in kTOKENIZER.tokenize(text) if word != '.']

class FeatureExtractor:
    def __init__(self): #, num_score_buckets, max_score1, max_score2):
        # self._max_score1 = max_score1
        # self._max_score2 = max_score2
        # self._num_score_buckets = num_score_buckets
        self._stopwords = nltk.corpus.stopwords.words('english')
        self.yago = yagoScores()
        self.wik = wikiscores()
        
        #yago.getScore(text,guessess)

    def features(self, entry, entry_score_distributions, combiner):
        i = 0
        #print (entry)
        for guess, guess_score_info in combiner.combine(entry_score_distributions):
            i+=1
            feature_dict = defaultdict(float)
            # self.add_features_from_question_text(feature_dict, entry)
            self.add_category_feature(feature_dict, entry)
            self.add_sentence_position_feature(feature_dict, entry)
            #print (str(guess) + "----" + str(guess_score_info))
            self.add_bucketized_score_features(feature_dict, combiner, guess_score_info)
            #self.add_yago_word_count_feature(feature_dict,entry,guess)
            #self.add_wiki_score(feature_dict,entry,guess)
            
            #print (str(feature_dict)+"--"+str(guess)+"--"+str(guess_score_info.combined_score()))
            #print feature_dict
            yield feature_dict, guess, guess_score_info.combined_score()
        #print (i)
    def add_sentence_position_feature(self, feature_dict, entry):
        feature_dict['sentence_position'] = entry['Sentence Position']

    def add_bucketized_score_features(self, feature_dict, combiner, guess_score_info):
        # i = 1
        # for score in scores:
        #     feature_dict['score_bucket_%s' % i] = combiner.bucketize_score(score, self._num_score_buckets)
        #     i += 1
        # for key, value in score_info.iteritems():
        #     # if key.startswith('bucket_score'):
        #     if key.startswith('bucket_norm_score'):
        #         feature_dict[key] = value
        #         # DEBUG
        #         # feature_dict['bucket_norm_score:main' if not 'bucket_norm_score:main' in feature_dict \
        #         #     else 'bucket_norm_score:secondary'] = value
        for type, score_info in guess_score_info.iteritems():
            #print (str(type)+"---"+str(score_info));
            feature_dict['Score bucket: %s' % type] = score_info.bucket_score()


    def add_features_from_question_text(self, feature_dict, entry):
        text = tokenize_and_stem(entry['Question Text'])

        self.add_word_count_features(feature_dict, text)
        # self.add_bigram_features(feature_dict, text)

    def add_word_count_features(self, feature_dict, tokenized_text):
        for word in tokenized_text:
            if word not in self._stopwords:
                feature_dict[word] += 1

    def add_bigram_features(self, feature_dict, tokenized_text):
        for first, second in nltk.bigrams(tokenized_text):
            if first not in self._stopwords and second not in self._stopwords:
                # TODO arbitrarily chosen feature weight
                feature_dict['bigram=(%s,%s)' % (first, second)] += 10

    def add_category_feature(self, feature_dict, entry):
        feature_dict['category'] = entry['category']

    # def add_guess_score_features(self, feature_dict, entry):
    #     for guess, score in ScoreCombiner(entry['QANTA Scores'], entry['IR_Wiki Scores']).iteritems():
    #         feature_dict['guess=' + guess] = int(round(score))
    def add_yago_word_count_feature(self, feature_dict, entry,guess):
        text = entry['Question Text']
        feature_dict['yagoCount'] =  self.yago.getScore(text,guess)
    
    def add_wiki_score(self,feature_dict,entry,guess):    
        if(int(entry['Sentence Position'])<2):
            text = entry['Question Text']
            feature_dict['WikiSearchFound'] = self.wik.getScore(text,guess)

class AnswerPredicter:
    kSCORE_TYPES = ['IR_Wiki Scores', 'QANTA Scores']

    def __init__(self, training_file, test_file, prediction_file, subsample, choose_max_sentences, debug):
        self._training_file = training_file
        self._test_file = test_file
        self._prediction_file = prediction_file
        self._subsample = subsample
        self._choose_max_sentences = choose_max_sentences
        self._debug = debug
        self._fe = FeatureExtractor()

        # self._max_score1, self._max_score2 = self.find_global_max_from_scores()

        self._combiner = ScoreCombiner(self.parse_all_scores(), 10)

    # def find_global_max_from_scores(self):
    #     qanta_scores = []
    #     ir_wiki_scores = []
    #     for entry in DictReader(open(self._training_file, 'r')):
    #         qanta_scores.extend(self.parse_scores(entry['QANTA Scores']))
    #         ir_wiki_scores.extend(self.parse_scores(entry['IR_Wiki Scores']))
    #
    #     # plt.hist(qanta_scores, bins=200)
    #     # plt.hist(ir_wiki_scores, bins=200)
    #     # plt.show()
    #     return max(qanta_scores), max(ir_wiki_scores)

    def execute_with_buckets(self, num_score_buckets):
        self._num_score_buckets = num_score_buckets

        print '\n========================='
        print 'Run with num_score_buckets = %s' % self._num_score_buckets
        start_time = time.time()

        accuracy = self.execute()

        print '\nCompleted in %0.5f seconds.' % (time.time() - start_time)

        return accuracy

    def execute(self):
        # fe = FeatureExtractor() #args.num_score_buckets, max_score1, max_score2)

        dev_train, dev_test, dev_test_entries = self.create_labeled_featuresets(self._fe)
        #print (dev_train)
        classifier = self.train_classifier(dev_train)
        accuracy = self.check_accuracy(classifier, dev_test_entries)
        
        self.error_analysis(classifier, dev_test_entries)
        
        if self._test_file:
            # Retrain on all data
            classifier = self.train_classifier(dev_train + dev_test)
            predictions = self.make_predictions_on_test_data(classifier)

        if self._prediction_file:
            self.write_predictions(predictions)

        return accuracy
        
    def error_analysis(self,classifier,entries):
        cats= ['history','lit','science','social']
        vals = ['wrong','right']
        
        right,wikiright,quantaright,bothright,bothwrong = 0,0,0,0,0
        
        err_file = DictWriter(open('errorAnalysis.csv', 'w'), ['Question ID','Question Text','Sentence Position','category','Answer','QANTA Scores','IR_Wiki Scores','Qanta Pos','Wiki Pos','Qanta CPos','Wiki CPos'])
        err_file.writeheader()  
        
        res_cats = defaultdict(int)
        res_pos = defaultdict(int)
        self._num_score_buckets = 34
        fe = FeatureExtractor()
        #dev_train, dev_test, dev_test_entries = self.create_labeled_featuresets(fe)
        #classifier = self.train_classifier(dev_train)

        for entry in entries:
            prediction = self.make_prediction(classifier,entry)
            ans = 0
            if prediction == entry['Answer']:
                right += 1
                ans = 1
            else:
                ansPos1 = -1
                ansPos2 = -1
                cPos1 = -1
                cPos2 = -1
                
                i = 0
                for each in entry['QANTA Scores'].split(","):
                    i += 1
                    #print each.split(":")[0]+"---"+entry["Answer"]
                    if(each.split(":")[0].strip() == entry["Answer"].strip()):    
                        ansPos1 = i
                    if(each.split(":")[0].strip() == prediction.strip()):    
                        cPos1 = i                        
                i = 0
                for each in entry['IR_Wiki Scores'].split(","):
                    i += 1
                    if(each.split(":")[0] == entry["Answer"]):    
                        ansPos2 = i                                
                    if(each.split(":")[0].strip() == prediction.strip()):    
                        cPos2 = i
                err_file.writerow({
                    'Question ID':entry['Question ID'],'Question Text':entry['Question Text'],'Sentence Position':entry['Sentence Position'],'category':entry['category'],'Answer':entry['Answer'],'QANTA Scores':entry['QANTA Scores'],'IR_Wiki Scores':entry['IR_Wiki Scores'],'Qanta Pos':ansPos1,'Wiki Pos':ansPos2,'Qanta CPos':cPos1,'Wiki CPos':cPos2})


            res_cats[entry['category'],vals[ans]]+=1
            pos=int(entry['Sentence Position']) 
            res_pos[pos,vals[ans]]+=1
            
        cat_percent = defaultdict(int)
        for each in cats:
            cat_percent[each] = float(res_cats[each,'right']) / float((res_cats[each,'right'] + res_cats[each,'wrong'])) 
        cat_percent=sorted(cat_percent.items(), key=itemgetter(1), reverse=True)
    
        pos_percent = defaultdict(int)
        for each in xrange(len(res_pos)/2):
            pos_percent[each] = float(res_pos[each,'right']) / float((res_pos[each,'right'] + res_pos[each,'wrong'])) 
        pos_percent=sorted(pos_percent.items(), key=itemgetter(1), reverse=True)
                
        print "\nAnalysis Sentence Pos:\n", res_pos
        print "\nAnalysis Categories  :\n",res_cats    
        print "\nAnalysis Cat. Percnt :\n",cat_percent
        print "\nAnalysis Sen.Pos Per :\n",pos_percent

    def write_predictions(self, predictions):
        output_file = DictWriter(open(self._prediction_file, 'w'), ['Question ID', 'Answer'])
        output_file.writeheader()

        for entry in sorted(predictions):
            output_file.writerow({
                'Question ID': entry,
                'Answer': predictions[entry][1]})

    # TODO Make top-level class so we can store member variables, such as max scores???
    def make_predictions_on_test_data(self, classifier):
        predictions = {}
        for entry in DictReader(open(self._test_file)):
            if(int(entry['Sentence Position'])<2):
                if(len(entry['QANTA Wiki'])>2):
                    prediction = entry['QANTA Wiki'].split(":")[0]
                    prediction = prediction.replace(",","") 
                    print prediction   
                elif(len(entry['Wiki Wiki'])>2):
                    prediction = entry['Wiki Wiki'].split(":")[0]
                    prediction = prediction.replace(",","")
                    print prediction
                else:
                    prediction = self.make_prediction(classifier, entry)
            else:
                prediction = self.make_prediction(classifier, entry)
                
            predictions[entry['Question ID']] = \
                (entry['Sentence Position'], self.make_prediction(classifier, entry))

        return predictions

    def make_prediction(self, classifier, entry):
        # combiner = ScoreCombiner(entry['QANTA Scores'], entry['IR_Wiki Scores'],
        #                          self._max_score1, self._max_score2, self._num_score_buckets)

        classifier_predictions = []
        entry_score_distributions = self.parse_entry_scores(entry)
        for features_for_guess, guess, combined_score in self._fe.features(entry, entry_score_distributions, self._combiner):
            guess_is_predicted_answer = classifier.classify(features_for_guess)
            if guess_is_predicted_answer:
                classifier_predictions.append((guess, combined_score))

        # DEBUG
        # assert len(classifier_predictions) > 0

        top_guesses = self._combiner.combine(entry_score_distributions)[:5]

        # Select final prediction from classifier prediction with max score, or top guess from combined list if classifier
        # fails to produce a guess.
        prediction = max(classifier_predictions, key=itemgetter(1))[0] if len(classifier_predictions) > 0 \
            else top_guesses[0][0]
        # prediction = max(classifier_predictions, key=itemgetter(1))[0] if len(classifier_predictions) > 0 \
        #     else 'BOGUS'

        self.debug_prediction(classifier_predictions, prediction, entry, top_guesses)

        return prediction

    def check_accuracy(self, classifier, entries):
        # debug('Labels of trained classifier: %s' % classifier.labels())

        right = 0
        total = len(entries)
        # for labeled_featureset, dev_test_entry in zip(test_features, dev_test_entries):
        for entry in entries:
            if(int(entry['Sentence Position'])<2):
                if(len(entry['QANTA Wiki'])>2):
                    prediction = entry['QANTA Wiki'].split(":")[0]
                    prediction = prediction.replace(",","") 
                    print prediction   
                elif(len(entry['Wiki Wiki'])>2):
                    prediction = entry['Wiki Wiki'].split(":")[0]
                    prediction = prediction.replace(",","")
                    print prediction
                else:
                    prediction = self.make_prediction(classifier, entry)
            else:
                prediction = self.make_prediction(classifier, entry)

            if prediction == entry['Answer']:
                right += 1

        accuracy = float(right) / float(total)
        print("Accuracy on dev: %f" % accuracy)

        return accuracy

    def debug_prediction(self, classifier_predictions, prediction, entry, top_guesses):
        # if not args.debug or len(classifier_predictions) == 1:
        if not self._debug or prediction == entry['Answer']:
            return

        # DEBUG
        if prediction == 'BOGUS':
            print 'Found bogus result, total so far:', counter.next()

        # debug('Answer:', labeled_featureset[1])
        self.debug('Answer:', entry['Answer'])
        # debug('Guess:', guess)
        self.debug('Classifier Predictions:', classifier_predictions)
        # debug('Classifier Prediction in Guesses?', classify_prediction in entry['QANTA Scores'] + entry['IR_Wiki Scores'])
        self.debug('Final Prediction:', prediction)
        self.debug('Prediction Correct?', prediction == entry['Answer'])

        self.debug('Question:', entry['Question Text'])
        self.debug('QANTA Guesses:', entry['QANTA Scores'])
        self.debug('IR Wiki Guesses:', entry['IR_Wiki Scores'])
        self.debug('Top Guesses:', top_guesses)
        # debug('Number of Features:', len(features))
        # debug('Features:\n', '\n'.join('\t%s: %s' % (key, value)
        #                                for key, value in sort_features_by_weight(features)))
        self.debug()

        # DEBUG
        # assert len(classifier_predictions) < 2

    def sort_features_by_weight(self, features):
        return sorted(features.items(), key=itemgetter(1), reverse=True)

    def train_classifier(self, features):
        print("Training classifier ...")
        return nltk.classify.DecisionTreeClassifier.train(features) #, verbose=True)
        # return nltk.classify.NaiveBayesClassifier.train(features)

    def create_labeled_featuresets(self, fe):
        dev_train = []
        dev_test = []
        dev_test_entries = []

        entries = DictReader(open(self._training_file, 'r'))
        if self._choose_max_sentences:
            entries = self.choose_max_sentences(entries)

        for entry in entries:
            if self._subsample < 1.0 and int(entry['Question ID']) % 1000 > 1000 * self._subsample:
                continue

            # combiner = ScoreCombiner(entry['QANTA Scores'], entry['IR_Wiki Scores'],
            #                  self._max_score1, self._max_score2, self._num_score_buckets)

            if int(entry['Question ID']) % 5 == 0:
                dev_test_entries.append(entry)
                self.append_labeled_featuresets(fe, dev_test, entry)
            else:
                self.append_labeled_featuresets(fe, dev_train, entry)

        return dev_train, dev_test, dev_test_entries

    def append_labeled_featuresets(self, fe, feature_list, entry):
        for features_for_guess, guess, score in fe.features(entry, self.parse_entry_scores(entry), self._combiner):
            #print features_for_guess
            feature_list.append((features_for_guess, guess == entry['Answer']))

    def choose_max_sentences(self, reader):
        filtered_entries = dict()
        for entry in reader:
            filtered_entries[entry['Question ID']] = entry

        return filtered_entries.values()

    # def parse_scores(self, score_string):
    #     return [float(guess_score.split(':')[1]) for guess_score in score_string.split(', ')]

    def parse_all_scores(self):
        score_dict = defaultdict(list)
        for entry in DictReader(open(args.training_file, 'r')):
            for type in self.kSCORE_TYPES:
                score_dict[type].extend(self.parse_scores(entry[type]).values())

        return score_dict

    def parse_scores(self, score_string):
        scores = dict()
        for guess, score in [guess_score.split(':') for guess_score in score_string.split(', ')]:
            scores[guess.strip()] = float(score)

        return scores

    def parse_entry_scores(self, entry):
        score_dict = defaultdict(list)
        for type in self.kSCORE_TYPES:
            score_dict[type] = self.parse_scores(entry[type])

        return score_dict

    def debug(self, *arguments):
        if self._debug:
            print ' '.join([str(argument) for argument in arguments])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify quiz bowl training data and predict answers for test data')
    parser.add_argument('training_file', help='input training dataset file')
    parser.add_argument('--test_file', help='input test dataset file - don\'t make predictions if unset')
    parser.add_argument('--prediction_file', help='output prediction file - don\'t write predictions if unset')

    # TODO subsample doesn't really do what we want right now, because the questions are repeated with increasing
    # sentence positions
    parser.add_argument('--subsample', type=float, default=1.0, help='subsample this amount')
    parser.add_argument('--choose_max_sentences', action='store_true', help='only consider version of each question '
                                                                            'with max sentence position')
    parser.add_argument('--num-score-buckets', type=int, help='number of buckets for normalized scores in decision tree')
    parser.add_argument('--debug', '-d', action='store_true', help='print verbose output for debugging')
    # TODO Add argument to output dev_train and dev_test entries, for debugging

    args = parser.parse_args()

    answer_predicter = AnswerPredicter(args.training_file, args.test_file, args.prediction_file, args.subsample,
                                       args.choose_max_sentences, args.debug)

    best_accuracy = 0.
    best_buckets = 0
    for buckets in [args.num_score_buckets] if args.num_score_buckets else xrange(24, 100):
        accuracy = answer_predicter.execute_with_buckets(buckets)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_buckets = buckets
            print 'This run had best accuracy found so far: %0.5f' % best_accuracy, 'buckets:', best_buckets

    print 'Final best accuracy: %0.5f' % best_accuracy, 'buckets:', best_buckets