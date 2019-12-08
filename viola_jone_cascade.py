from viola_jone import ViolaJones
import math
from tqdm import tqdm
import numpy as np
import pickle
import os 
dir_path = os.path.abspath('')

class ViolaJonesCascade:
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training, testing, load_feature=''):
        pos, neg = [], []
        for ex in training: 
            if ex[1] == 1:
                pos.append(ex)
            else:
                neg.append(ex)
        for feature_num in self.layers:
            if len(neg) == 0:
                print("FPR = 0, stop the training!")
                break
            clf = ViolaJones(T=feature_num, v=False)
            clf.train(training, testing, load_feature=load_feature)
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            print('# of non-face photos abandoned : ', len(neg) - len(false_positives))
            neg = false_positives
        self.test(testing)
        
    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1

    def test(self, testing):
        total = len(testing)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for image, yVal in testing:
            pred = self.classify(image)
            if pred == yVal:
                if yVal == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if yVal == 1:
                    FN += 1
                else:
                    FP += 1
        
        print('Testing:')
        print('Total accuracy Rate: {} ({}/{})'.format((TP + TN) / total, TP + TN, total))
        print('False Positive Rate: {} ({}/{})'.format(FP / (FP + TN), FP, FP + TN))
        print('False Negative Rate: {} ({}/{}) \n'.format(FN / (FN + TP), FN, FN + TP))