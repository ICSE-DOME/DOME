#!/usr/bin/env python
#
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>
# modified: 2021-12-1 elloworl

import numpy as np


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    '''

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, comment_pred, comment):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param comment_pred: list : list tokens of one test sentence
        :param comment: list : list tokens of the corresponding reference sentences
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        prec = []
        rec = []

        for reference in comment:
            # compute the longest common subsequence
            lcs = my_lcs(reference, comment_pred)
            prec.append(lcs / float(len(comment_pred)))
            rec.append(lcs / float(len(reference)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, ids, comment_pred, comment):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param ids: list : the id of the reference sentences
        :param comment_pred: list : list tokens of candidate / test sentences
        :param comment: list : list tokens of reference sentences
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert len(ids) == len(comment_pred) == len(comment)

        score = dict()
        for i, id in enumerate(ids):
            score[id] = self.calc_score(comment_pred[i], comment[i])

        average_score = np.mean(np.array(list(score.values())))
        return average_score, score

    def method(self):
        return "Rouge"
