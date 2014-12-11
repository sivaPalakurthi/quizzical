from csv import DictReader, DictWriter
import argparse

from score_combiner import ScoreCombiner
from collections import defaultdict

import matplotlib.pyplot as plt

class AnswerPredicter:
    kSCORE_TYPES = ['IR_Wiki Scores', 'QANTA Scores']

    def __init__(self, training_file, test_file, prediction_file, debug):
        self._training_file = training_file
        self._test_file = test_file,
        self._prediction_file = prediction_file
        self._debug = debug

        self._combiner = ScoreCombiner(self.parse_all_scores(), 10)

    def parse_all_scores(self):
        score_dict = defaultdict(list)
        for entry in DictReader(open(args.training_file, 'r')):
            for type in self.kSCORE_TYPES:
                score_dict[type].extend(self.parse_scores(entry[type]).values())

        return score_dict

    # TODO Combine methods?
    def parse_entry_scores(self, entry):
        score_dict = defaultdict(list)
        for type in self.kSCORE_TYPES:
            score_dict[type] = self.parse_scores(entry[type])

        return score_dict

    def parse_scores(self, score_string):
        scores = dict()
        for guess, score in [guess_score.split(':') for guess_score in score_string.split(', ')]:
            scores[guess.strip()] = float(score)

        return scores

    def execute(self):
        # DEBUG Plot normalized full score distributions
        # print self._combiner._score_stats
        #
        # for type, stats in self._combiner._score_stats.iteritems():
        #     plt.hist([value / stats.stddev() for value in stats.distribution()], bins=300)
        #
        # plt.show()

        self.check_accuracy(self.retrieve_dev_test_entries())

        if args.test_file:
            predictions = self.make_predictions_on_test_data()

        if args.prediction_file:
            self.write_predictions(predictions)

    def write_predictions(self, predictions):
        output_file = DictWriter(open(args.prediction_file, 'w'), ['Question ID', 'Answer'])
        output_file.writeheader()

        for entry in sorted(predictions):
            output_file.writerow({
                'Question ID': entry,
                'Answer': predictions[entry][1]})

    def make_predictions_on_test_data(self):
        predictions = {}
        for entry in DictReader(open(args.test_file)):
            predictions[entry['Question ID']] = \
                (entry['Sentence Position'], self.make_prediction(entry))

        return predictions

    # TODO Fix this
    def make_prediction(self, entry):
        # return self._combiner.combine(self.parse_entry_scores(entry)).sorted_list()[0][0]
        return self._combiner.combine(self.parse_entry_scores(entry))[0][0]

    def check_accuracy(self, entries):
        right = 0
        current = 0
        total = len(entries)
        for entry in entries:
            current += 1
            prediction = self.make_prediction(entry)
            if prediction == entry['Answer']:
                right += 1

            print("Processed %s of %s entries..." % (current, total))

        print("Accuracy on dev: %f" % (float(right) / float(total)))

    def retrieve_dev_test_entries(self):
        entries = []
        for entry in DictReader(open(args.training_file, 'r')):
            if int(entry['Question ID']) % 5 == 0:
                entries.append(entry)

        return entries

    def debug(self, *arguments):
        if args.debug:
            print ' '.join([str(argument) for argument in arguments])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify quiz bowl training data and predict answers for test data')
    parser.add_argument('training_file', help='input training dataset file')
    parser.add_argument('--test_file', help='input test dataset file - don\'t make predictions if unset')
    parser.add_argument('--prediction_file', help='output prediction file - don\'t write predictions if unset')
    parser.add_argument('--debug', '-d', action='store_true', help='print verbose output for debugging')

    args = parser.parse_args()

    AnswerPredicter(args.training_file, args.test_file, args.prediction_file, args.debug).execute()

