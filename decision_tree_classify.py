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

kTOKENIZER = TreebankWordTokenizer()

# DEBUG
counter = count()

def morphy_stem(word):
    stem = wn.morphy(word)
    return stem.lower() if stem else word.lower()

def tokenize_and_stem(text):
    return [morphy_stem(word.strip('.')) for word in kTOKENIZER.tokenize(text) if word != '.']

class FeatureExtractor:
    def __init__(self, num_score_buckets):
        self._num_score_buckets = num_score_buckets
        self._stopwords = nltk.corpus.stopwords.words('english')

    def features(self, entry):
        combiner = ScoreCombiner(entry['QANTA Scores'], entry['IR_Wiki Scores'])
        for guess, scores, combined_score in combiner.iteritems():
            feature_dict = defaultdict(float)
            # self.add_features_from_question_text(feature_dict, entry)
            self.add_category_feature(feature_dict, entry)
            self.add_bucketized_score_feature(feature_dict, combiner, scores)

            yield feature_dict, guess, scores

    def add_bucketized_score_feature(self, feature_dict, combiner, scores):
        i = 1
        for score in scores:
            feature_dict['score_bucket_%s' % i] = combiner.bucketize_score(score, self._num_score_buckets)
            i += 1

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

def main():
    fe = FeatureExtractor(args.num_score_buckets)

    dev_train, dev_test, dev_test_entries = create_labeled_featuresets(fe)
    classifier = train_classifier(dev_train)
    check_accuracy(fe, classifier, dev_test_entries)

    if args.test_file:
        # Retrain on all data
        classifier = train_classifier(dev_train + dev_test)
        predictions = make_predictions_on_test_data(fe, classifier)

    if args.prediction_file:
        write_predictions(predictions)

def write_predictions(predictions):
    output_file = DictWriter(open(args.prediction_file, 'w'), ['Question ID', 'Answer'])
    output_file.writeheader()

    for entry in sorted(predictions):
        output_file.writerow({
            'Question ID': entry,
            'Answer': predictions[entry][1]})

def make_predictions_on_test_data(fe, classifier):
    predictions = {}
    for entry in DictReader(open(args.test_file)):

        predictions[entry['Question ID']] = \
            (entry['Sentence Position'], make_prediction(fe, classifier, entry))

    return predictions

def make_prediction(fe, classifier, entry):
    classifier_predictions = []
    for features_for_guess, guess, score in fe.features(entry):
        guess_is_predicted_answer = classifier.classify(features_for_guess)
        if guess_is_predicted_answer:
            classifier_predictions.append((guess, score))

    # DEBUG
    # assert len(classifier_predictions) > 0

    top_guesses = ScoreCombiner(entry['QANTA Scores'], entry['IR_Wiki Scores']).sorted_list()[:5]

    # Select final prediction from classifier prediction with max score, or top guess from combined list if classifier
    # fails to produce a guess.
    prediction = max(classifier_predictions, key=itemgetter(1))[0] if len(classifier_predictions) > 0 \
        else top_guesses[0][0]
    # prediction = max(classifier_predictions, key=itemgetter(1))[0] if len(classifier_predictions) > 0 \
    #     else 'BOGUS'
    # prediction = top_guesses[0][0]

    debug_prediction(classifier_predictions, prediction, entry, top_guesses)

    return prediction

def check_accuracy(fe, classifier, entries):
    # debug('Labels of trained classifier: %s' % classifier.labels())

    right = 0
    total = len(entries)
    # for labeled_featureset, dev_test_entry in zip(test_features, dev_test_entries):
    for entry in entries:
        prediction = make_prediction(fe, classifier, entry)

        if prediction == entry['Answer']:
            right += 1

    print("Accuracy on dev: %f" % (float(right) / float(total)))

def debug_prediction(classifier_predictions, prediction, entry, top_guesses):
    # if not args.debug or len(classifier_predictions) == 1:
    if not args.debug or prediction == entry['Answer']:
        return

    # DEBUG
    if prediction == 'BOGUS':
        print 'Found bogus result, total so far:', counter.next()

    # debug('Answer:', labeled_featureset[1])
    debug('Answer:', entry['Answer'])
    # debug('Guess:', guess)
    debug('Classifier Predictions:', classifier_predictions)
    # debug('Classifier Prediction in Guesses?', classify_prediction in entry['QANTA Scores'] + entry['IR_Wiki Scores'])
    debug('Final Prediction:', prediction)
    debug('Prediction Correct?', prediction == entry['Answer'])

    debug('Question:', entry['Question Text'])
    debug('QANTA Guesses:', entry['QANTA Scores'])
    debug('IR Wiki Guesses:', entry['IR_Wiki Scores'])
    debug('Top Guesses:', top_guesses)
    # debug('Number of Features:', len(features))
    # debug('Features:\n', '\n'.join('\t%s: %s' % (key, value)
    #                                for key, value in sort_features_by_weight(features)))
    debug()

    # DEBUG
    # assert len(classifier_predictions) < 2

def sort_features_by_weight(features):
    return sorted(features.items(), key=itemgetter(1), reverse=True)

def train_classifier(features):
    print("Training classifier ...")
    return nltk.classify.DecisionTreeClassifier.train(features) #, verbose=True)
    # return nltk.classify.NaiveBayesClassifier.train(features)

def create_labeled_featuresets(fe):
    dev_train = []
    dev_test = []
    dev_test_entries = []

    entries = DictReader(open(args.training_file, 'r'))
    if args.choose_max_sentences:
        entries = choose_max_sentences(entries)

    for entry in entries:
        if args.subsample < 1.0 and int(entry['Question ID']) % 1000 > 1000 * args.subsample:
            continue

        if int(entry['Question ID']) % 5 == 0:
            dev_test_entries.append(entry)
            append_labeled_featuresets(fe, dev_test, entry)
        else:
            append_labeled_featuresets(fe, dev_train, entry)

    return dev_train, dev_test, dev_test_entries

def append_labeled_featuresets(fe, feature_list, entry):
    for features_for_guess, guess, score in fe.features(entry):
        feature_list.append((features_for_guess, guess == entry['Answer']))

def choose_max_sentences(reader):
    filtered_entries = dict()
    for entry in reader:
        filtered_entries[entry['Question ID']] = entry

    return filtered_entries.values()

def parse_scores(score_string):
    return [float(guess_score.split(':')[1]) for guess_score in score_string.split(', ')]

def debug(*arguments):
    if args.debug:
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
    parser.add_argument('--num-score-buckets', type=int, default=5, help='number of buckets for normalized scores in '
                                                                         'decision tree')
    parser.add_argument('--debug', '-d', action='store_true', help='print verbose output for debugging')
    # TODO Add argument to output dev_train and dev_test entries, for debugging

    args = parser.parse_args()

    start_time = time.time()

    main()

    print '\nCompleted in %0.5f seconds.' % (time.time() - start_time)