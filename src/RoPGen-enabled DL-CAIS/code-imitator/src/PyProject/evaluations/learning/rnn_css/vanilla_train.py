# NOTE: This script is for training a vanilla CCS18 model on datasets other than the original C++ one,
# or for training a data-augmented, or PGD CCS18 model on all datasets.
# To train a GradAug CCS18 model, use `gradaug_train.py`.
# To train a vanilla model on the original C++ dataset, refer to the README of the original project.

import sys
import os

# FIXME #1: Change this to your project root.
repo_path = '/home/hx/data/code-imitator/'
sys.path.insert(1, os.path.join(repo_path, 'src', 'PyProject'))

from featureextractionV2.StyloFeaturesProxy import StyloFeaturesProxy
from featureextractionV2.StyloFeatures import StyloFeatures
from featureextractionV2.StyloUnigramFeatures import StyloUnigramFeatures
from classification import StratifiedKFoldProblemId
from featureextractionV2 import utils_extraction

import numpy as np
import copy

from ConfigurationLearning.ConfigurationLearningRNN import ConfigurationLearningRNN
import ConfigurationGlobalLearning as Config
from classification.NovelAPI.Learning import Learning
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
from keras import backend as K
import keras.optimizers
import keras.callbacks


# config = tf.ConfigProto(device_count = {'GPU': 0})
# session = tf.Session(config=config)
# K.set_session(session)

# FIXME #7: Select the GPU you want to use. Set to -1 to disable GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

np.set_printoptions(threshold=sys.maxsize)
############ Input
# parser = argparse.ArgumentParser(description='Start Attack')
# parser.add_argument('problemid', type=str, nargs=1,
#                    help='the problem id')
# parser.add_argument('gpu', type=str, nargs=1,
#                    help='the gpu to be used')
# args = parser.parse_args()
# PROBLEM_ID_LOADED = args.problemid[0]
# PROBLEM_ID_LOADED = "3264486_5736519012712448"
# GPU_LOADED = args.gpu[0]
# GPU_LOADED = "1"
# print("Loaded:", PROBLEM_ID_LOADED, " with GPU:", GPU_LOADED)


# Further parameters:
# we use the following dataset.
#datasetpath = os.path.join(repo_path, "data", "train")

# FIXME #2: Change this to your training set.
trainnewpath = '/home/hx/data/code-imitator/data/trainnew_augBaseline_offline_githubc'
#trainnewpath = os.path.join(repo_path, "data", "gradaug_cpp_mini_aug2_3prime")
#trainnewpath = '/home/hx/data/code-imitator/data/gradaug_cpp_test'
#trainnewpath = '/home/hx/data/code-imitator/data/dataset_2017/dataset_train_githubc/fold5'
#trainnewpath = '/home/hx/data/code-imitator/data/trainnew_mcts_cpp_direct_aug3'
#trainnewpath = '/home/hx/data/code-imitator/data/train_gcjjava_fold2'
#trainnewpath = '/home/hx/data/code-imitator/data/train_cppgcj_fold3'
#trainnewpath = '/home/hx/data/code-imitator/data/train_java40_fold2'
#trainnewpath = '/home/hx/data/code-imitator/data/trainnew_gcjjava1517_fold1_direct_aug3'

# FIXME #3: Change this to your testing set (could be arbitrary, doesn't affect training results). Technically, the testing set is of no use
# throughout this script, but the method that trains the RF classifier (i.e., compute_rlf_on_rnn(), which was written by the MCTS attack authors)
# requires a testing set in order to compute the accuracy, so better make it happy.
testpath = os.path.join(repo_path, "data", "testset")
prexmlpath = os.path.join(repo_path, "data", "prexml")
tfxmlpath = os.path.join(repo_path, "data", "tfxml")
stylexmlpath = os.path.join(repo_path, "data", "stylexml")
# we specify some stop words, see ConfigurationGlobalLearning.py
stop_words_codestylo = ["txt", "in", "out", "attempt0", "attempt", "attempt1", "small", "output", "input"]
# We assume 8 files per author
probsperprogrammer = 8
# we assume a dataset of 204 authors in total
no_programmers = 204

############### Variable Definition ##############

config_learning: ConfigurationLearningRNN = ConfigurationLearningRNN(
    repo_path=Config.repo_path,
    dataset_features_dir=os.path.join(Config.repo_path, "data/dataset_2017"),
    suffix_data="_2017_8_formatted_macrosremoved",
    learnmodelspath=Config.learnmodelspath,
    use_lexems=False,
    use_lexical_features=False,
    stop_words=Config.stop_words_codestylo,
    probsperprogrammer=Config.probsperprogrammer,
    no_of_programmers = 204,
    noofparallelthreads=8,
    scale=True,
    cv_optimize_rlf_params=False,
    cv_use_rnn_output=False,
    hyperparameters=None
)


threshold_sel: int = 800
learn_method: str = "RNN"

learning: Learning = Learning()
threshold = 800

from MySequence import MySequence
import classification.utils_learning_rnn as utils_learning_rnn
feature_dim = threshold

# FIXME #4: Change this to the number of authors in the training set.
num_classes = 67
print(num_classes)
#if config_learning.hyperparameters is None:

# FIXME #5: Change 'RNN_epochs' to the max number of training epochs. Note that the epoch number starts at 0.
param_grid = {
                "RNN_epochs": [1001], #350], #50],
                "RNN_nounits": [288], #, feature_dim],
                "RNN_dropout": [0.6],
                "RNN_lstmlayersno": [3],
                "RNN_denselayersno": [3],
                "RNN_l2reg": [0.00001],
                "RNN_denseneurons": [round(0.45*feature_dim)]
                }
#else:
#    param_grid = config_learning.hyperparameters
#    param_grid['RNN_denseneurons'] = [round(x * feature_dim) for x in param_grid['RNN_denseneurons']]

if config_learning.cv_optimize_rlf_params:
    param_grid_rf = {"RF_n_estimators": [250],
                    "RF_max_features": [0.3, 0.6, 'auto'],
                    "RF_max_depth": [10, 25, 50, 75, None],
                    "RF_min_samples_leaf": [6, 12, 1],
                    }
    param_grid.update(param_grid_rf)
# kerasclf = KerasClassifier(build_fn=utils_learning_rnn.my_model, batch_size=128, input_dim_eq=feature_dim, output_dim_eq=num_classes,
#                         optimizer="Adam", verbose=0)

# FIXME #6: This was a grid search in the original version of code. For time reasons, we got rid of that and manually set the params,
# so if you want the grid search back, you'll have to change this.
best_params_, best_params_acc = [params for params in ParameterGrid(param_grid)][0], None
best_params_rnn, best_params_rf = utils_learning_rnn.split_params_into_rnn_rf(params=best_params_)

early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=20, verbose=1, min_delta=0.0)
# param['callbacks'] = [early_stop]
# param['validation_data'] = (x_test.reshape(x_test.shape[0], feature_dim, 1), y_test_c)

# C. Learn on best params
# FIXME #8: Change this to the batch size you want.
seq = MySequence(trainnewpath, testpath, prexmlpath, tfxmlpath, stylexmlpath, batch_size=128, n=12)

# FIXME #8: Change this to the batch size you want.
clf_best = KerasClassifier(build_fn=utils_learning_rnn.my_model, batch_size=128,
                            input_dim_eq=feature_dim, output_dim_eq=num_classes,
                            optimizer=keras.optimizers.Adam(lr=10e-4),
                            callbacks=[],
                            verbose=True)
clf_best.set_params(**best_params_rnn)


model = utils_learning_rnn.my_model(**clf_best.filter_sk_params(utils_learning_rnn.my_model))
seq.set_model(model)
fit_args = copy.deepcopy(clf_best.filter_sk_params(Sequential.fit_generator))
seq.set_graph(tf.get_default_graph())

rnnhist = model.fit_generator(seq, **fit_args)