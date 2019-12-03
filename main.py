from viola_jone import ViolaJones
from PIL import Image
import argparse
import numpy as np
import glob
import pickle
import os 

parser = argparse.ArgumentParser(description='ViolaJones Algorithm')
parser.add_argument('-T', help='# of rounds', type=int, default=10)
parser.add_argument('-criterion', help='Criterion for model optimization', type=str, default='err', choices=['err', 'fpr', 'fnr'])
parser.add_argument('-load_feat', help='Load precomputed features file', type=str, default='')
parser.add_argument('-width', help='Maximal width of feature', type=int, default=8)
parser.add_argument('-height', help='Maximal height of feature',type=int, default=8)

args = vars(parser.parse_args())


dir_path = os.path.abspath('')
types_name = {1:'Two Vertical', 2:'Two Horizontal', 3:'Three Horizontal', 4:'Three Vertical', 5:'Four'}
folders = {'non-faces' : 0, 'faces' : 1}

trainData = []
for folder, yVal in folders.items():
    for filename in glob.glob('./dataset/trainset/' + folder + '/*.png'):
        im = Image.open(filename)
        trainData.append((np.asarray(im, dtype="float32") / 255, yVal))
        
testData = []
for folder, yVal in folders.items():
    for filename in glob.glob('./dataset/testset/' + folder + '/*.png'):
        im = Image.open(filename)
        testData.append([np.asarray(im, dtype="float32") / 255, yVal])

print('# of training images : ', len(trainData))
print('# of testing images : ', len(testData))


def save_model(model, file_name):
    with open(os.path.join(dir_path, file_name), 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def read_mode(file_name):
    with open(os.path.join(dir_path, file_name), 'rb') as input:
        model = pickle.load(input)
        return model
# test_trainData = trainData[0:100] + trainData[2300:]
# test_testData = testData[0:100] + testData[2100:]
model = ViolaJones(T=args['T'])
model.train(trainData, testData, args['height'], args['width'], crit=args['criterion'], load_feature=args['load_feat'])

if not os.path.exists(os.path.join(dir_path, './save_models')):
    os.makedirs(os.path.join(dir_path, './save_models'))

print('save trained model at {}'.format('./save_models/model_' + args['criterion'] + '_' + str(args['T'])))
save_model(model, './save_models/model_' + args['criterion'] + '_' + str(args['T']))


# model.train(test_trainData, test_testData, args['height'], args['width'], crit=args['criterion'], load_feature=args['load_feat'])