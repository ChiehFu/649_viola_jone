import math
from tqdm import tqdm
import numpy as np
from utils import RectangleRegion, integral_image
import pickle
import os 
dir_path = os.path.abspath('')

class ViolaJones:
    
#     class Feature(Enum):
#         type_1 = 'Two Vertical'
#         type_2 = 'Two Horizontal'
#         type_3 = 'Three Horizontal'
#         type_4 = 'Three Vertical'
#         type_5 = 'Four'
        
    class WeakClassifier:
        def __init__(self, haar_feature, threshold, polarity):
            self.haar_feature = haar_feature
            self.threshold = threshold
            self.polarity = polarity
            self.acc = None
            
        def classify(self, x):
            return 1 if self.polarity * self.haar_feature.compute_features(x) < self.polarity * self.threshold else 0
    
    class HaarFeature:
        def __init__(self, harr_type, positive_regions, negative_regions, position, width, height):
            self.harr_type = harr_type
            self.positive_regions = positive_regions
            self.negative_regions = negative_regions
            self.position = position
            self.width = width
            self.height = height
            
        def compute_features(self, x):
            return sum([pos.compute_feature(x) for pos in self.positive_regions]) - sum([neg.compute_feature(x) for neg in self.negative_regions])
        
            
    def __init__(self, T, v=True):
        self.T = T
        self.alphas = []
        self.clfs = []
        self.test_acc = []
        self.test_fpr = []
        self.test_fnr = []
        self.verbose = v
    def build_features(self, image_shape, max_height, max_width, load_feature=''):

        height, width = image_shape
        
        features = []
        type_count = [0] * 5
        
        if max_height == None:
            max_height = height
        else:
            max_height = min(max_height, height)
            
        if max_width == None:
            max_width = width
        else:
            max_width = min(max_width, width)
            
        for w in range(1, max_width + 1):
            for h in range(1, max_height + 1):
                cur_x = 0
                while cur_x + w <= width:
                    cur_y = 0
                    
                    while cur_y + h <= height:
                        
                        rec = RectangleRegion(cur_x, cur_y, w, h)

                        if cur_x + 2 * w <=  width:
                            # type 1 (two vertical) features
                            rec_right = RectangleRegion(cur_x + w, cur_y, w, h)
                            features.append(self.HaarFeature(1, [rec], [rec_right], (cur_x, cur_y), 2 * w, h))
                            type_count[0] += 1
                            
                            # type 4 (three vertical) features.
                            if cur_x + 3 * w <= width:
                                rec_right_right = RectangleRegion(cur_x + 2 * w, cur_y, w, h)
                                features.append(self.HaarFeature(4, [rec, rec_right_right], [rec_right], (cur_x, cur_y), 3 * w, h))
                                type_count[3] += 1
                                
                        # type 2 (two horizontal) features.
                        if cur_y + 2 * h <= height:
                            rec_bot = RectangleRegion(cur_x, cur_y + h, w, h)
                            features.append(self.HaarFeature(2, [rec], [rec_bot], (cur_x, cur_y), w, 2 * h))
                            type_count[1] += 1
                            
                            # type 3 (three horizontal) features.
                            if cur_y + 3 * h <= height:
                                rec_bot_bot = RectangleRegion(cur_x, cur_y + 2 * h, w, h)
                                features.append(self.HaarFeature(3, [rec, rec_bot_bot], [rec_bot], (cur_x, cur_y), w, 3 * h))
                                type_count[2] += 1
                                
                        # type 5 (four) features.
                        if (cur_x + 2 * w <=  width) and (cur_y + 2 * h <= height):
                            rec_bot_right = RectangleRegion(cur_x + w, cur_y + h, w, h)
                            features.append(self.HaarFeature(5, [rec, rec_bot_right], [rec_right, rec_bot], (cur_x, cur_y), 2 * w, 2 * h))
                            type_count[4] += 1
                            
                        cur_y += 1
                    cur_x += 1

        if self.verbose:            
            # Print the feature summary 
            print('\t The total number of Haar Features is : ', len(features))
            print('\t There are ', type_count[0] ,' type 1 (two vertical) features.')
            print('\t There are ', type_count[1],' type 2 (two horizontal) features.')
            print('\t There are ', type_count[2],' type 3 (three horizontal) features.')
            print('\t There are ', type_count[3],' type 4 (three vertical) features.')
            print('\t There are ', type_count[4],' type 5 (four) features.')

        return features


    def apply_features(self, features, training_data, load_feature=''):
        y = np.array(list(map(lambda data: data[1], training_data)))

        if load_feature != '':
            print('Load calculated features...')
            with open(os.path.join(dir_path, load_feature), 'rb') as input:
                X = pickle.load(input)
                return X, y

        X = np.zeros((len(features), len(training_data)))
        i = 0
        for haar_feature in tqdm(features):
            X[i] = list(map(lambda data: haar_feature.compute_features(data[0]), training_data))
            i += 1


        if not os.path.exists(os.path.join(dir_path, './save_features/')):
            os.makedirs(os.path.join(dir_path, './save_features/'))

        print('Save precomputed feature at {}'.format(os.path.join(dir_path, 'save_features/features_{}'.format(len(features)))))
        with open(os.path.join(dir_path, 'save_features/features_{}'.format(len(features))), 'wb') as output:
            pickle.dump(X, output, pickle.HIGHEST_PROTOCOL)

        return X, y

    def train_weak(self, X, y, features, weights):
        total_pos, total_neg = 0, 0

        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        
        pbar = tqdm(total=total_features)
        
        for index, feature in enumerate(X):
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf = self.WeakClassifier(best_feature, best_threshold, best_polarity)
            classifiers.append(clf)
            
            pbar.update(1)
            
        return classifiers
        
    def select_best(self, classifiers, weights, training_data, crit='err'):
        # Pick the best classifier by emperical error, false positive rate, or false negative rate
        best_clf, best_err, best_acc = None, float('inf'), None
        best_fp_err, best_fn_err = float('inf'), float('inf')
        for clf in classifiers:
            err, acc = 0, []
            fp_err, fn_err = 0, 0
            
            for data, w in zip(training_data, weights):
                
                pred = clf.classify(data[0])
                c = abs(pred - data[1])
                acc.append(c)
                err += w * c
                
                if pred != data[1]:
                    if data[1] == 1:
                        fn_err += w
                    else:
                        fp_err += w
            err = err / len(training_data)   

            if crit == 'err' and err < best_err:
                best_clf, best_err, best_acc = clf, err, acc
            elif crit == 'fpr' and fp_err < best_fp_err:
                best_clf, best_err, best_acc = clf, err, acc
                best_fp_err = fp_err
            elif crit == 'fnr' and fn_err < best_fn_err:
                best_clf, best_err, best_acc = clf, err, acc
                best_fn_err = fn_err

        # Set training accuracy for the selected classifier
        best_clf.acc = sum(1 if i == 0 else 0 for i in best_acc) / len(best_acc)
        return best_clf, best_err, best_acc
    
    def train(self, training, testing, max_height=8, max_width=8, crit='err', load_feature=''):
        
        pos_num = sum([tup[1] for tup in training])
        neg_num = len(training) - pos_num
        print('Face / None-Face : {}/{}'.format(pos_num, neg_num))
        rule = 'emperical error' if crit=='err' else 'false positive rate' if crit=='fpr' else 'false negative rate'
        print('Seleting classfiers based on', rule)
        weights = np.zeros(len(training))
        training_data = []
        
        print('Initialize the weights of {} weak classfiers...'.format(self.T))
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)
                
        print('Build up Haar features filter of the size {}x{}'.format(max_height, max_width))
        features = self.build_features(training_data[0][0].shape, max_height, max_width)
        
        print('Precompute the Haar features of the training set...')
        X, y = self.apply_features(features, training_data, load_feature)
#         print(X.shape)
#         print(y.shape)
        print('Start Adaboost...')
    
        for t in range(self.T):
            print('Round {}/{}:'.format(t + 1, self.T))
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, total_error, accuracy = self.select_best(weak_classifiers, weights, training_data, crit)
            
            beta = total_error / (1.0 - total_error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            # if crit == 'err' or crit == 'fpr':
            #     beta = total_error / (1.0 - total_error)
            #     for i in range(len(accuracy)):
            #         weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            #     alpha = math.log(1.0/beta)
            # elif crit == 'fnr':
            #     alpha = math.log((1.0 - total_error) /total_error) *  (1 - fp_error) / fn_error
            #     for i in range(len(accuracy)):
            #         weights[i] = weights[i] * math.exp(alpha * accuracy[i])

            self.alphas.append(alpha)
            self.clfs.append(clf)
            
            if t == self.T - 1:
                self.test(testing, t + 1)
            
    def classify(self, image):
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def test(self, testing, t):
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
        
        self.test_acc.append((TP + TN) / total)
        self.test_fpr.append(FP / (FP + TN))
        self.test_fnr.append(FN / (FN + TP))
        
        print('Testing at Round {} :'.format(t))
        print('Total accuracy Rate: {} ({}/{})'.format((TP + TN) / total, TP + TN, total))
        print('False Positive Rate: {} ({}/{})'.format(FP / (FP + TN), FP, FP + TN))
        print('False Negative Rate: {} ({}/{}) \n'.format(FN / (FN + TP), FN, FN + TP))
        