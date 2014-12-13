__author__ = 'Patrick'

from collections import defaultdict
from math import sqrt
from operator import itemgetter
import numpy

class ScoreStats:
    def __init__(self, distribution):
        self._distribution = distribution
        np_array = numpy.array(distribution)

        self._mean = np_array.mean()
        self._stddev = np_array.std()
        self._max = np_array.max()

    def distribution(self):
        return self._distribution

    def mean(self):
        return self._mean

    def stddev(self):
        return self._stddev

    def max(self):
        return self._max

    def __repr__(self):
        return 'Mean: %s, Std dev: %s' % (self._mean, self._stddev)

class GuessScoreInfo:
    def __init__(self, guess, stddev_cutoff, num_buckets, score_stats_dict):
        self._guess = guess
        self._stddev_cutoff = stddev_cutoff
        self._num_buckets = num_buckets
        self._type_dict = dict()
        self._score_stats_dict = score_stats_dict

    def guess(self):
        return self._guess

    def incorporate_score(self, type, score):
        self._type_dict[type] = ScoreInfo(type, score, self._stddev_cutoff, self._num_buckets,
                                          self._score_stats_dict[type])

    def combined_score(self):
        # return self._sqrt_sum_of_squares([score_info.norm_score() for score_info in self._type_dict.values()])
        # return max([score_info.norm_score() for score_info in self._type_dict.values()])
        return self._sqrt_sum_of_squares([score_info.norm_score() + self._magic_offset(type) for type, score_info in
                                          self._type_dict.iteritems()])
        # return self._sqrt_sum_of_squares([max(0, score_info.norm_score()) for type, score_info in
        #                                   self._type_dict.iteritems()])

    def _magic_offset(self, type):
        return 0.1 if type == 'QANTA Scores' else 3

    def get_score_info(self, type):
        return self._type_dict[type]

    def iteritems(self):
        return self._type_dict.iteritems()

    def _sqrt_sum_of_squares(self, values):
        return sqrt(sum([value ** 2 for value in values]))

class ScoreInfo:
    kBUCKET_BEYOND_CUTOFF = 100

    def __init__(self, type, score, stddev_cutoff, num_buckets, score_stats):
        self._type = type
        self._score = score
        self._norm_score = self._calculate_norm_score(score_stats)
        self._bucket_score = self._calculate_bucket_score(stddev_cutoff, num_buckets, score_stats)

    def _calculate_norm_score(self, score_stats):
        # BEST SCORE: 0.768675
        # return ((self._score - score_stats.mean()) / score_stats.stddev()) + 0.1 * score_stats.stddev()

               # (0.01 if self._type == 'QANTA Scores' else 3.0445)
        # return ((self._score - score_stats.mean()) / score_stats.stddev()) + \
        #        (0.1 if self._type == 'QANTA Scores' else 3)

        return (self._score - score_stats.mean()) / score_stats.stddev()

    def score(self):
        return self._score

    def norm_score(self):
        return self._norm_score

    def bucket_score(self):
        return self._bucket_score

    def _calculate_bucket_score(self, stddev_cutoff, num_buckets, score_stats):
        # return (int) (num_buckets * self.score() / score_stats.max())
        # return (int) (self._norm_score)

        if abs(self._norm_score) > stddev_cutoff:
            return (-1 if self._norm_score < 0 else 1) * self.kBUCKET_BEYOND_CUTOFF

        return (int) (round(self._norm_score * (num_buckets / 2.) / stddev_cutoff))

class ScoreCombiner:
    # kNORMALIZED_MAX = 100.

    def __init__(self, score_distributions, stddev_cutoff, num_buckets):
        self._stddev_cutoff = stddev_cutoff
        self._num_buckets = num_buckets

        self._score_stats = dict()
        for type, scores in score_distributions.iteritems():
            self._score_stats[type] = ScoreStats(scores)

    def combine(self, entry_score_distributions):
        guesses = dict()
        for type, score_dict in entry_score_distributions.iteritems():
            for guess, score in score_dict.iteritems():
                if not guess in guesses:
                    guesses[guess] = GuessScoreInfo(guess, self._stddev_cutoff, self._num_buckets, self._score_stats)
                guesses[guess].incorporate_score(type, score)

        sorted_list = self.sorted_guess_list(guesses)
        return sorted_list

    def sorted_guess_list(self, guesses):
        return sorted([(guess, guess_score_info) for guess, guess_score_info in guesses.iteritems()],
                      key=lambda item: item[1].combined_score(), reverse=True)

    # def _calculate_combined_scores(self):
    #     for guess in self._score_dict.keys():
    #         self._score_dict[guess]['combined_score'] = \
    #             self._sqrt_sum_of_squares(self._score_dict[guess][label] for label in self._score_dict[guess]
    #                                       if label.startswith('norm_score'))
    #
    # def format_top_scores(self, top_number=5, join_string='\n\t'):
    #     return join_string + \
    #            join_string.join(['%s: %s' % (label, score) for label, score in self.sorted_list()[:top_number]])

    def num_buckets(self):
        return self._num_buckets

    def stddev_cutoff(self):
        return self._stddev_cutoff

    def iteritems(self):
        return self._score_dict.iteritems()
