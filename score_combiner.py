__author__ = 'Patrick'

from collections import defaultdict
from math import sqrt
from operator import itemgetter

class ScoreCombiner:
    def __init__(self, score_string1, score_string2):
        self._score_dict = defaultdict(float)
        self._combine(score_string1, score_string2)

    def _combine(self, score_string1, score_string2):
        for label, score in self._build_normalized_score_dict(score_string1).items() + \
                            self._build_normalized_score_dict(score_string2).items():
            self._incorporate_score(label, score)

    def sorted_list(self):
        return sorted(self._score_dict.items(), key=itemgetter(1), reverse=True)

    def _incorporate_score(self, label, score):
        self._score_dict[label] = score if label not in self._score_dict else \
            self._sqrt_sum_of_squares(self._score_dict[label], score)

    def _sqrt_sum_of_squares(self, value1, value2):
        return sqrt(value1 ** 2 + value2 ** 2)

    def _build_normalized_score_dict(self, score_string):
        return self._normalize_scores(dict([(key.strip(), float(value)) for key, value in
                [label_score_pairs.split(":") for label_score_pairs in score_string.split(", ")]]))

    def _normalize_scores(self, score_dict):
        max_score = max(score_dict.values())
        if max_score == 0:
            return dict()
        normalizer = 100. / max_score

        for key in score_dict:
            score_dict[key] *= normalizer

        return score_dict

    def format_top_scores(self, top_number=5, join_string='\n\t'):
        return join_string + \
               join_string.join(['%s: %s' % (label, score) for label, score in self.sorted_list()[:top_number]])

    def iteritems(self):
        return self._score_dict.iteritems()
