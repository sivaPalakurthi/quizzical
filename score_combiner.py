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
    def __init__(self, guess, num_buckets, score_stats_dict):
        self._guess = guess
        self._num_buckets = num_buckets
        self._type_dict = dict()
        self._score_stats_dict = score_stats_dict

    def guess(self):
        return self._guess

    def incorporate_score(self, type, score):
        self._type_dict[type] = ScoreInfo(type, score, self._num_buckets, self._score_stats_dict[type])

    def combined_score(self):
        return self._sqrt_sum_of_squares([score_info.norm_score() for score_info in self._type_dict.values()])

    def get_score_info(self, type):
        return self._type_dict[type]

    def iteritems(self):
        return self._type_dict.iteritems()

    def _sqrt_sum_of_squares(self, values):
        return sqrt(sum([value ** 2 for value in values]))

class ScoreInfo:
    # kSHIFT_N_STDDEV = 2

    def __init__(self, type, score, num_buckets, score_stats):
        self._type = type
        self._score = score
        self._norm_score = self._calculate_norm_score(score_stats)
        self._bucket_score = self._calculate_bucket_score(num_buckets, score_stats)

    def _calculate_norm_score(self, score_stats):
        # BEST SCORE: 0.768675
        # return ((self._score - score_stats.mean()) / score_stats.stddev()) + 0.1 * score_stats.stddev()

               # (0.01 if self._type == 'QANTA Scores' else 3.0445)
        return ((self._score - score_stats.mean()) / score_stats.stddev()) + \
               (0.1 if self._type == 'QANTA Scores' else 3)

    def score(self):
        return self._score

    def norm_score(self):
        return self._norm_score

    def bucket_score(self):
        return self._bucket_score

    def _calculate_bucket_score(self, num_buckets, score_stats):
        # return (int) (num_buckets * self.score() / score_stats.max())
        return (int) (self._norm_score)

class ScoreCombiner:
    kNORMALIZED_MAX = 100.

    def __init__(self, score_distributions, num_buckets):
        # self._max_score1 = max_score1
        # self._max_score2 = max_score2
        self._num_buckets = num_buckets

        # self._score_dict = defaultdict(dict)
        # self.combine(score_string1, score_string2, max_score1, max_score2)

        self._score_stats = dict()
        for type, scores in score_distributions.iteritems():
            self._score_stats[type] = ScoreStats(scores)

    def combine(self, entry_score_distributions):
        # for label, score in self._build_normalized_score_dict(score_string1, max_score1).items() + \
        #                     self._build_normalized_score_dict(score_string2).items():
        #     self._incorporate_score(label, score)

        # for type, distribution in entry_score_distributions.iteritems():
        #     self._add_scores_to_dict(distribution, type)
            # self._add_scores_to_dict(score_string2, max_score2, '2')

        # self._calculate_combined_scores()

        guesses = dict()
        for type, score_dict in entry_score_distributions.iteritems():
            # if type == 'IR_Wiki Scores':
            # if type == 'QANTA Scores':
            for guess, score in score_dict.iteritems():
                if not guess in guesses:
                    guesses[guess] = GuessScoreInfo(guess, self._num_buckets, self._score_stats)
                guesses[guess].incorporate_score(type, score)

        sorted_list = self.sorted_guess_list(guesses)
        return sorted_list

    def sorted_guess_list(self, guesses):
        return sorted([(guess, guess_score_info) for guess, guess_score_info in guesses.iteritems()],
                      key=lambda item: item[1].combined_score(), reverse=True)

    def _calculate_combined_scores(self):
        for guess in self._score_dict.keys():
            self._score_dict[guess]['combined_score'] = \
                self._sqrt_sum_of_squares(self._score_dict[guess][label] for label in self._score_dict[guess]
                                          if label.startswith('norm_score'))

    # def sorted_list(self):
    #     return sorted(self._score_dict.items(), reverse=True,
    #                   key=lambda score_item: score_item[1]['combined_score'])

    # def _incorporate_score(self, label, score):
    #     # self._score_dict[label] = score if label not in self._score_dict else \
    #     #     self._sqrt_sum_of_squares(self._score_dict[label], score)
    #     self._score_dict[label]['norm_scores'].append(score)

    # def _sqrt_sum_of_squares(self, values):
    #     return sqrt(sum([value ** 2 for value in values]))

    # def _add_scores_to_dict(self, score_dict, score_type):
    #     for guess, score in score_dict.iteritems():
    #         self._score_dict[guess]['original_score:%s' % score_type] = score
    #         # self._score_dict[guess]['bucket_score:%s' % type_label] = self.bucketize_score(score, max_score)
    #
    #     for guess, norm_score in self._build_normalized_score_dict(score_string, score_type).iteritems():
    #         self._score_dict[guess]['norm_score:%s' % score_type] = norm_score
    #         self._score_dict[guess]['bucket_norm_score:%s' % score_type] = self.bucketize_score(norm_score, self.kNORMALIZED_MAX)

    # def _build_normalized_score_dict(self, score_string, type_label):
    #     return self._normalize_scores(dict([(key.strip(), float(value)) for key, value in
    #             [label_score_pairs.split(":") for label_score_pairs in score_string.split(", ")]]), type_label)
    #
    # def _normalize_scores(self, score_dict, type_label):
    #     # max_score = max(score_dict.values())
    #     # if max_score == 0:
    #     #     return dict()
    #     # normalizer = self.kNORMALIZED_MAX / max_score
    #
    #     for key in score_dict:
    #         score_dict[key] /= self._score_stats[type_label].stddev()
    #
    #     return score_dict

    # def bucketize_score(self, type, score_info):
    #     # return (int) (num_buckets * score / self._sqrt_sum_of_squares(self.kNORMALIZED_MAX, self.kNORMALIZED_MAX))
    #     return (int) (self._num_buckets * score_info.norm_score() / self._score_stats[type].max())

    def format_top_scores(self, top_number=5, join_string='\n\t'):
        return join_string + \
               join_string.join(['%s: %s' % (label, score) for label, score in self.sorted_list()[:top_number]])

    def iteritems(self):
        # for field, value in self._score_dict.iteritems():
        #     yield field, value, self._sqrt_sum_of_squares(value)
        return self._score_dict.iteritems()
