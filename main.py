from viola_jone import ViolaJones
from PIL import Image
import argparse
import numpy as np
import glob
import pickle
import os 

parser = argparse.ArgumentParser(description='ViolaJones Algorithm')
parser.add_argument('-t', help='# of rounds', type=int, default=10)
parser.add_argument('-c', help='Criterion for model optimization', type=str, default='err', choices=['err', 'fpr', 'fnr'])
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

model = ViolaJones(T=args['t'])
model.train(trainData, testData, 8, 8, crit=args['c'])
save_model(model, 'model_' + args['c'])