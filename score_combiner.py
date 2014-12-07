__author__ = 'Patrick'

from collections import defaultdict
from math import sqrt
from operator import itemgetter

class ScoreCombiner:
    kNORMALIZED_MAX = 100.

    def __init__(self, score_string1, score_string2, max_score1, max_score2, num_buckets):
        self._max_score1 = max_score1
        self._max_score2 = max_score2
        self._num_buckets = num_buckets

        self._score_dict = defaultdict(dict)
        self._combine(score_string1, score_string2, max_score1, max_score2)

    def _combine(self, score_string1, score_string2, max_score1, max_score2):
        # for label, score in self._build_normalized_score_dict(score_string1, max_score1).items() + \
        #                     self._build_normalized_score_dict(score_string2).items():
        #     self._incorporate_score(label, score)
        self._add_scores_to_dict(score_string1, max_score1, '1')
        self._add_scores_to_dict(score_string2, max_score2, '2')

        self._calculate_combined_scores()

    def _calculate_combined_scores(self):
        for guess in self._score_dict.keys():
            self._score_dict[guess]['combined_score'] = \
                self._sqrt_sum_of_squares(self._score_dict[guess][label] for label in self._score_dict[guess]
                                          if label.startswith('norm_score'))

    def sorted_list(self):
        return sorted(self._score_dict.items(), reverse=True,
                      key=lambda score_item: score_item[1]['combined_score'])

    def _incorporate_score(self, label, score):
        # self._score_dict[label] = score if label not in self._score_dict else \
        #     self._sqrt_sum_of_squares(self._score_dict[label], score)
        self._score_dict[label]['norm_scores'].append(score)

    def _sqrt_sum_of_squares(self, values):
        return sqrt(sum([value ** 2 for value in values]))

    def _add_scores_to_dict(self, score_string, max_score, type_label):
        for guess, score in [(guess.strip(), float(score)) for guess, score in [label_score_pairs.split(":") for \
                label_score_pairs in score_string.split(", ")]]:
            self._score_dict[guess]['original_score:%s' % type_label] = score
            self._score_dict[guess]['bucket_score:%s' % type_label] = self.bucketize_score(score, max_score)

        for guess, norm_score in self._build_normalized_score_dict(score_string).iteritems():
            self._score_dict[guess]['norm_score:%s' % type_label] = norm_score
            self._score_dict[guess]['bucket_norm_score:%s' % type_label] = self.bucketize_score(norm_score, self.kNORMALIZED_MAX)

    def _build_normalized_score_dict(self, score_string):
        return self._normalize_scores(dict([(key.strip(), float(value)) for key, value in
                [label_score_pairs.split(":") for label_score_pairs in score_string.split(", ")]]))

    def _normalize_scores(self, score_dict):
        max_score = max(score_dict.values())
        if max_score == 0:
            return dict()
        normalizer = self.kNORMALIZED_MAX / max_score

        for key in score_dict:
            score_dict[key] *= normalizer

        return score_dict

    def bucketize_score(self, score, max_score):
        # return (int) (num_buckets * score / self._sqrt_sum_of_squares(self.kNORMALIZED_MAX, self.kNORMALIZED_MAX))
        return (int) (self._num_buckets * score / max_score)

    def format_top_scores(self, top_number=5, join_string='\n\t'):
        return join_string + \
               join_string.join(['%s: %s' % (label, score) for label, score in self.sorted_list()[:top_number]])

    def iteritems(self):
        # for field, value in self._score_dict.iteritems():
        #     yield field, value, self._sqrt_sum_of_squares(value)
        return self._score_dict.iteritems()
