import os

from keras.utils import Sequence
import sys
import math
import importlib
import copy
import subprocess
import numpy as np
import pickle
import json
import shutil

import threading
import multiprocessing
import random
import string
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from featureextractionV2.StyloFeaturesProxy import StyloFeaturesProxy
from featureextractionV2.StyloFeatures import StyloFeatures
from featureextractionV2.StyloUnigramFeatures import StyloUnigramFeatures
from classification import StratifiedKFoldProblemId
from featureextractionV2 import utils_extraction
from classification.NovelAPI.Learning import Learning

from timeit import default_timer as timer
from ctypes import *

from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from keras.losses import categorical_crossentropy
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid
import classification.utils_learning_rnn as utils_learning_rnn


from keras.utils import to_categorical

from ConfigurationLearning.ConfigurationLearningRNN import ConfigurationLearningRNN
import ConfigurationGlobalLearning as Config
import classification.NovelAPI.utils_classification

from distutils.dir_util import copy_tree

#from for_while import trans_tree

#from lxml import etree
ns = {'src': 'http://www.srcML.org/srcML/src',
    'cpp': 'http://www.srcML.org/srcML/cpp',
    'pos': 'http://www.srcML.org/srcML/position'}
doc = None

# FIXME #9: Change this to the number of authors in the training set.
num_classes = 67
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
    hyperparameters={
                      "RNN_epochs": [300], #350], #50],
                      "RNN_nounits": [288], #, feature_dim],
                      "RNN_dropout": [0.6],
                      "RNN_lstmlayersno": [3],
                      "RNN_denselayersno": [3],
                      "RNN_l2reg": [0.00001],
                      "RNN_denseneurons": [0.45]
                      }
)

def ce(y_true, y_pred, eps=1e-15):
    y_pred /= y_pred.sum(axis=-1)[:, np.newaxis]
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=-1)

def remove_dir_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def init_parser(file):
    global doc
    parser = etree.XMLParser(encoding='gbk')
    doc = etree.parse(file, parser=parser)
    e = etree.XPathEvaluator(doc, namespaces=ns)
    return e

def cmd(command, wait=True):
    if not wait:
        subp = subprocess.Popen(command,shell=False)
        return
    subp = subprocess.Popen(command,shell=True)
    subp.wait(2)
    if subp.poll() == 0:
        pass
    #    print("success!")
    else:
        print("fail")

def save_learnsetup(modelsavedir: str, curproblem: str, learn_method: str, threshold,
                      testlearnsetup):
    """
    Small helper function to save the testlearnsetup to disk.
    """

    if modelsavedir is not None:
        modelfile = os.path.join(modelsavedir,
                                 "model_" + curproblem + "_" + learn_method + "_" + str(threshold) + ".pck")
        pickle.dump(testlearnsetup, file=open(modelfile, 'wb'))


def save_keras_model(modelsavedir: str, curproblem: str, learn_method: str, threshold,
                       keras_model, keras_hist):
    if modelsavedir is not None:
        kerasmodelpath = os.path.join(modelsavedir, "keras_model_" + curproblem
                                      + "_" + learn_method + "_" + str(threshold) + ".pck")
        keras_model.save(kerasmodelpath)
        if keras_hist is not None:
            histfile = os.path.join(modelsavedir, "keras_model_hist_" +
                                    curproblem + "_" + learn_method + "_" + str(threshold) + ".pck")
            pickle.dump(keras_hist, file=open(histfile, 'wb'))

class MySequence(Sequence):
    def __init__(self, trainpath, testpath, prexmlpath, tfxmlpath, stylexmlpath, batch_size=128, n=None, filterbydict_orig_model=None):
        self.trainpath, self.testpath, self.prexmlpath, self.tfxmlpath, self.stylexmlpath = trainpath, testpath, prexmlpath, tfxmlpath, stylexmlpath
        self.numeric_tf_path = './transformations/numeric'
        self.list_tf_path = './transformations/list'
        self.len = sum(len(files) for _, _, files in os.walk(os.path.join(self.trainpath, '0')))
        self.batch_size = batch_size
        self.next_path = ''
        self.epoch = 0
        self.real_epoch = 0
        self.x, self.y = None, None
        self.graph = None
        self.is_checkpoint = False
        self.n = n
        self.filterbydict_orig_model = filterbydict_orig_model
        self.libsrcml = None
        self.is_kickstarted = False
        self.random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 5))
        # old_tfxml = self.tfxmlpath
        self.tfxmlpath += '_' + self.random_str
        if not os.path.exists(self.tfxmlpath):
            os.makedirs(self.tfxmlpath)
            copy_tree(self.prexmlpath, self.tfxmlpath)

        feature_dim = 800
        param_grid = {
                      "RNN_epochs": [901], #350], #50],
                      "RNN_nounits": [288], #, feature_dim],
                      "RNN_dropout": [0.6],
                      "RNN_lstmlayersno": [3],
                      "RNN_denselayersno": [3],
                      "RNN_l2reg": [0.00001],
                      "RNN_denseneurons": [round(0.45*feature_dim)]
                      }
        best_params_, best_params_acc = [params for params in ParameterGrid(param_grid)][0], None
        self.best_params_rnn, self.best_params_rf = utils_learning_rnn.split_params_into_rnn_rf(params=best_params_)


    def set_model(self, model):
        self.model = model

    def set_graph(self, graph):
        self.graph = graph

    def set_callback(self, callback):
        self.callback = callback

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def preprocess(self, feature_vec):
        if self.sc is not None:
            featvec = self.sc.transform(feature_vec[0, :].todense())
        else:
            featvec = np.array(feature_vec[0, :].todense())

        featvec = featvec.reshape(featvec.shape[0], 1, featvec.shape[1])
        #featvec_deep = self.get_deep_features([featvec, 0])[0]
        return featvec

    def save_model(self, modelsavedir, curproblem, rlf):
        from classification.LearnSetups.LearnSetupRNNRF import LearnSetupRNNRF
        testlearnsetup = LearnSetupRNNRF(data_final_train=self.train_obj, data_final_test=None,
                                             clf=None, rlf=rlf, stdscaler=self.sc)
        threshold = 800
        if not os.path.exists(modelsavedir):
            os.makedirs(modelsavedir)
        save_learnsetup(modelsavedir=modelsavedir, curproblem=curproblem, learn_method='RNN',
                              threshold=threshold, testlearnsetup=testlearnsetup)
        save_keras_model(modelsavedir=modelsavedir, curproblem=curproblem, learn_method='RNN',
                               threshold=threshold, keras_model=self.model, keras_hist=None)

    def srcml_convert(self, lock, src, dst):
        with lock:
            self.libsrcml.srcml(bytes(src, encoding='utf8'), bytes(dst, encoding='utf8'))

    def transform_numeric(self, lock, e, tree, root, transform, transformed_author_path, inter_epoch_ignore_dict, intra_epoch_ignore_dict, author, prog, transformed_progs):
        #prog_style = transform.get_program_style(e)
        possible_styles = transform.get_possible_styles()
        # instances_and_styles = transform.get_instances_and_styles(e, tree)
        instance_idx = 0
        tf_prog_cnt = 0
        # ignore_list = inter_epoch_ignore_dict.get((author, prog, transform.__name__), [])
        ignore_list = []
        # for item in instances_and_styles:
        #     path_list = item[0]
        #     style_type = item[1]
        #     style_value = item[2]
        style_value = ''
        path_list = []
        for value in possible_styles:
            # if value != style_value:
            tree_copy = copy.deepcopy(root).getroottree()
            e_copy = etree.XPathEvaluator(tree_copy, namespaces=ns)
            
            transformed_xml_path = os.path.abspath(os.path.join(transformed_author_path, '_'.join([prog, transform.__name__, str(instance_idx), str(value) + '.xml'])))
            transformed_cpp_path = os.path.abspath(os.path.join(transformed_author_path, '_'.join([prog, transform.__name__, str(instance_idx), str(value) + '.cpp'])))

            per_tf_ignore_list = transform.transform(e_copy, path_list, style_value, value, transformed_xml_path, ignore_list)
            if per_tf_ignore_list:
                #print('per_tf_ignore_list', per_tf_ignore_list)
                tmp = intra_epoch_ignore_dict.get((author, prog, transform.__name__), [])
                with lock:
                    intra_epoch_ignore_dict[(author, prog, transform.__name__)] = tmp + per_tf_ignore_list

                #cmdstr = ["srcml", transformed_xml_path, "-o", transformed_cpp_path]
                #cmd(cmdstr, False)
                self.srcml_convert(lock, transformed_xml_path, transformed_cpp_path)
                tf_prog_cnt += 1
                transformed_cpp_name = os.path.basename(transformed_cpp_path)
                with lock:
                    transformed_progs[transformed_cpp_name] = (prog, transform.__name__, per_tf_ignore_list, author)
        instance_idx += 1
        return tf_prog_cnt

    def transform_list(self, lock, transform, transformed_author_path, orig_train_dir, no_of_samples, author, prog, orig_prog_path, prog_fullpath, inter_epoch_ignore_dict, intra_epoch_ignore_dict, transformed_progs):
        authors = os.listdir(orig_train_dir)
        target_authors = random.sample(authors, no_of_samples)
        # ignore_list = inter_epoch_ignore_dict.get((author, prog, transform.__name__), [])
        ignore_list = []
        for target_author in target_authors:
            if target_author == author: continue
            target_author_path = os.path.join(self.prexmlpath, target_author)
            transformed_xml_path = os.path.abspath(os.path.join(transformed_author_path, '_'.join([prog, transform.__name__, author, target_author + '.xml'])))
            transformed_cpp_path = os.path.abspath(os.path.join(transformed_author_path, '_'.join([prog, transform.__name__, author, target_author + '.cpp'])))
            per_tf_ignore_list = transform.transform(prog_fullpath, target_author_path, orig_prog_path, transformed_xml_path, ignore_list)
            if per_tf_ignore_list:
                tmp = intra_epoch_ignore_dict.get((author, prog, transform.__name__), [])
                with lock:
                    intra_epoch_ignore_dict[(author, prog, transform.__name__)] = tmp + per_tf_ignore_list
                #cmdstr = ["srcml", transformed_xml_path, "-o", transformed_cpp_path]
                #cmd(cmdstr, False)
                self.srcml_convert(lock, transformed_xml_path, transformed_cpp_path)
                transformed_cpp_name = os.path.basename(transformed_cpp_path)
                with lock:
                    transformed_progs[transformed_cpp_name] = (prog, transform.__name__, per_tf_ignore_list, author)

    def make_next_dataset_worker(self, lock, author, num_of_files, num_of_instances, orig_train_dir, transformed_progs, intra_epoch_ignore_dict, inter_epoch_ignore_dict):
        for prog in os.listdir(os.path.join(self.trainpath, str(self.epoch), author)):
            numeric_tfs = [importlib.import_module('transformations.numeric.' + tf_name.split('.')[:-1][0]) \
                        for tf_name in os.listdir(self.numeric_tf_path) if tf_name.endswith('.py')]
            list_tfs = [importlib.import_module('transformations.list.' + tf_name.split('.')[:-1][0]) \
                        for tf_name in os.listdir(self.list_tf_path) if tf_name.endswith('.py')]
            
            with lock:
                num_of_files[0] += 1
            prog_fullpath = os.path.join(self.tfxmlpath, author, '.'.join(prog.split('.')[:1]) + '.xml')
            # orig_prog_path = os.path.join(self.trainpath, str(self.epoch), author, prog)
            orig_prog_path = os.path.join(self.trainpath, '0', author, prog)
            prog_stripped = prog.replace('.cpp', '')
            transformed_author_path = os.path.join('.', 'transformed_' + self.random_str, author, prog_stripped)
            with lock:
                if not os.path.exists(transformed_author_path):
                    os.makedirs(transformed_author_path)
                shutil.copy(orig_prog_path, transformed_author_path)
                print(orig_prog_path, transformed_author_path)
                transformed_progs[prog] = (prog, 'orig', [], author)
            # cmdstr = "srcml \""+orig_prog_path+"\" -o \""+prog_fullpath+"\""
            # cmd(cmdstr)
            #self.srcml_convert(lock, orig_prog_path, prog_fullpath)
            # e = init_parser(prog_fullpath)
            # root = e('/*')[0]
            # tree = root.getroottree()
            # for transform in numeric_tfs:
            #     try:
            #         ret = self.transform_numeric(lock, e, tree, root, transform, transformed_author_path, inter_epoch_ignore_dict, intra_epoch_ignore_dict, author, prog, transformed_progs)
            #         with lock:
            #             num_of_instances[0] += ret
            #     except:
            #         print('Error!!!!!!!!!!!!', orig_prog_path, prog_fullpath)
            #         import traceback
            #         traceback.print_exc()
            #         with open('error.txt', 'a') as f:
            #             f.write(orig_prog_path)
            #             f.write(traceback.format_exc())
            # for transform in list_tfs:
            #     try:
            #         self.transform_list(lock, transform, transformed_author_path, orig_train_dir, 2, author, prog, orig_prog_path, prog_fullpath, inter_epoch_ignore_dict, intra_epoch_ignore_dict, transformed_progs)
            #     except:
            #         print('Error!!!!!!!!!!!!', orig_prog_path, prog_fullpath)
            #         import traceback
            #         traceback.print_exc()
            #         with open('error.txt', 'a') as f:
            #             f.write(orig_prog_path)
            #             f.write(traceback.format_exc())

            max_loss = 0
            max_prog = 0
            max_author = 0
            
            learning: Learning = Learning()
            threshold = 800
            stop_words_codestylo = ["txt", "in", "out", "attempt0", "attempt", "attempt1", "small", "output", "input"]
            tmp_path = os.path.join('.', 'transform_tmp_' + self.random_str, author, prog_stripped)
            tmp_author_path = os.path.join('.', 'transform_tmp_' + self.random_str, author, prog_stripped, prog_stripped)
            try:
                os.makedirs(tmp_author_path)
            except OSError:
                pass
            # select_path = os.path.join(self.trainpath, 'select', author)
            # if len(os.listdir(select_path)) < 10:
            #     cpy_src = os.path.join(self.trainpath, str(self.epoch), author, prog)
            #     cpy_dst = os.path.join(self.trainpath, str(self.epoch+1), author, prog)
            #     try:
            #         os.makedirs(os.path.join(self.trainpath, str(self.epoch+1), author))
            #     except OSError:
            #         pass
            #     # print(cpy_src, cpy_dst)
            #     shutil.copy(cpy_src, cpy_dst)
            #     remove_dir_contents(transformed_author_path)
            #     remove_dir_contents(tmp_path)
            #     continue
            # for adv_prog in os.listdir(select_path):
            #     adv_prog_path = os.path.join(select_path, adv_prog)
            #     if prog.split('_')[:2] == adv_prog.split('_')[:2]:
            #         shutil.copy(adv_prog_path, transformed_author_path)
            #         transformed_progs[adv_prog] = (prog, adv_prog, [], author)
            copy_tree(transformed_author_path, tmp_author_path)

            unigrammmatrix_test = StyloUnigramFeatures(inputdata=tmp_path,
                                                        nocodesperprogrammer=len(os.listdir(tmp_author_path)),
                                                        noprogrammers=1,
                                                        binary=False, tf=True, idf=True,
                                                        ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                                        trainobject=self.unigrammmatrix_train)
            testfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=unigrammmatrix_test)
            #listoftraintestsplits = learning.do_local_train_test_split(train_obj=self.trainfiles, config_learning=config_learning,
            #                                                             threshold=threshold, trainproblemlength=None)

            modelsavedir = config_learning.learnmodelspath
            curproblem = '000'
            learn_method = '000'
            modelfile = os.path.join(modelsavedir,
                                         "model_" + curproblem + "_" + learn_method + "_" + str(threshold) + ".pck")
            kerasmodelpath = os.path.join(modelsavedir, "keras_model_" + curproblem
                                              + "_" + learn_method + "_" + str(threshold) + ".pck")

            with self.graph.as_default():
                testfiles.createtfidffeatures(trainobject=self.unigrammmatrix_train)
                testfiles.selectcolumns(index=None, trainobject=self.trainfiles)
                print(">Whole train set: TRAIN:", self.trainfiles.getfeaturematrix().shape, "TEST:", testfiles.getfeaturematrix().shape)
                # import time
                # time.sleep(2)

                x_train, y_train = self.trainfiles.getfeaturematrix(), self.trainfiles.getlabels()
                x_test = testfiles.getfeaturematrix()
                y_test = testfiles.getlabels()
                authors = testfiles.getauthors()
                doclabels = testfiles.getdoclabels()

                max_loss = 0
                max_prog = ''
                max_author = ''
                for i in range(x_test.shape[0]):
                    y_true = to_categorical(y_test[i], num_classes=num_classes)
                    x_test_scaled = self.preprocess(x_test[i])
                    y_pred_proba = self.model.predict_proba(x_test_scaled)[0]
                    #print(y_pred_proba)
                    loss = ce(np.array([y_true]), np.array([y_pred_proba]))[0]
                    #print('predicted:', y_pred_proba)
                    if loss > max_loss:
                        max_loss = loss
                        max_prog = doclabels[i]
                        max_author = authors[i]

                print(max_author, max_prog)
            max_info = transformed_progs.get(max_prog, None)
            if max_info is None:
                max_prog = prog
                max_author = author
            max_author_path = os.path.join(self.trainpath, str(self.epoch+1), transformed_progs[max_prog][3])
            try:
                os.makedirs(max_author_path)
            except OSError:
                pass
            replace_src = os.path.join('.', 'transformed_' + self.random_str, transformed_progs[max_prog][3], max_author, max_prog)
            replace_dst = os.path.join(max_author_path, transformed_progs[max_prog][0])
            if os.path.exists(replace_dst): os.remove(replace_dst)
            print(replace_src, replace_dst)
            shutil.copyfile(replace_src, replace_dst)

            # k = (max_author, transformed_progs[max_prog][0], transformed_progs[max_prog][1])
            # tmp = inter_epoch_ignore_dict.get(k, [])
            # with lock:
            #     inter_epoch_ignore_dict[k] = tmp + transformed_progs[max_prog][2]
            #     with open('inter_epoch_ignore_dict.pkl', 'wb') as f:
            #         pickle.dump(inter_epoch_ignore_dict, f)

            remove_dir_contents(transformed_author_path)
            remove_dir_contents(tmp_path)

    def make_next_dataset(self, patience=2):
        if self.libsrcml is None:
            self.libsrcml = cdll.LoadLibrary('/usr/lib/libsrcml.so.1')
        orig_train_dir = os.path.join(self.trainpath, '0')
        this_train_dir = os.path.join(self.trainpath, str(self.epoch+1))
        if self.epoch == 0:
            # remove_dir_contents(os.path.join('.', 'transform_tmp'))
            # remove_dir_contents(os.path.join('.', 'transformed'))
            if os.path.exists('inter_epoch_ignore_dict.pkl'): os.remove('inter_epoch_ignore_dict.pkl')
        # elif self.n is not None and self.epoch % self.n == 0:
        #     print('Epoch ', self.epoch + 1, ' resetting training set...')
        #     copy_tree(orig_train_dir, this_train_dir)
        #     if os.path.exists('inter_epoch_ignore_dict.pkl'): os.remove('inter_epoch_ignore_dict.pkl')
        #     return -1
        author_list = os.listdir(orig_train_dir)
        transformed_progs = {}
        if not os.path.exists('inter_epoch_ignore_dict.pkl'):
            inter_epoch_ignore_dict = {}
        else:
            with open('inter_epoch_ignore_dict.pkl', 'rb') as f:
                inter_epoch_ignore_dict = pickle.load(f)
        intra_epoch_ignore_dict = {}
        num_of_files = [0]
        num_of_instances = [0]
        m = multiprocessing.Manager()
        lock = m.Lock()
        with ThreadPoolExecutor(max_workers=1) as e:
            for author in author_list:
                threads = []
                args = (lock, author, num_of_files, num_of_instances, orig_train_dir, transformed_progs, intra_epoch_ignore_dict, inter_epoch_ignore_dict)
                t = e.submit(self.make_next_dataset_worker, *args)
                threads.append(t)

            for t in threads:
                t.result()

        if self.n is None:
            self.n = num_of_instances[0] // num_of_files[0]
        print('n is now ', self.n)
        return 0

    def get_next_dataset(self):
        try:
            if self.x is None:
                if self.filterbydict_orig_model is None:
                    learning: Learning = Learning()
                    threshold = 800

                    stop_words_codestylo = ["txt", "in", "out", "attempt0", "attempt", "attempt1", "small", "output", "input"]
                    # self.next_path = os.path.join(self.trainpath, str(self.epoch))
                    self.next_path = os.path.join(self.trainpath, '0')
                    self.unigrammmatrix_train = StyloUnigramFeatures(inputdata=self.next_path,
                                                        nocodesperprogrammer=8,
                                                        noprogrammers=len(os.listdir(self.next_path)),
                                                        binary=False, tf=True, idf=True,
                                                        ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                                        trainobject=None)
                    self.trainfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=self.unigrammmatrix_train)
                    # FIXME #10: Change 'noprogrammers' below to the number of authors in your testing set.
                    unigrammmatrix_test = StyloUnigramFeatures(inputdata=self.testpath,
                                                                nocodesperprogrammer=1,
                                                                noprogrammers=1,
                                                                binary=False, tf=True, idf=True,
                                                                ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                                                trainobject=self.unigrammmatrix_train)
                    testfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=unigrammmatrix_test)

                    print(">Whole train set: TRAIN:", self.trainfiles.getfeaturematrix().shape, "TEST:", testfiles.getfeaturematrix().shape)
                    self.train_obj, test_obj = learning._tfidf_feature_selection(train_obj=self.trainfiles, test_obj=testfiles,
                                                                        config_learning=config_learning,
                                                                   threshold=threshold)
                else:
                    self.train_obj, test_obj = self.filterbydict_orig_model.data_final_train, self.filterbydict_orig_model.data_final_test

                x_train, self.y_train = self.train_obj.getfeaturematrix(), self.train_obj.getlabels()
                if test_obj is not None:
                    x_test = test_obj.getfeaturematrix()
                    self.y_test = test_obj.getlabels()
                else:
                    x_test = None
                    self.y_test = None

                import preprocess
                # keep 'num_classes' in sync with the one in test.py
                self.sc, x_train, self.x_test, y_train_c, y_test_c, feature_dim = preprocess.preprocess(num_classes, config_learning, x_train, x_test, self.y_train, self.y_test)

                self.x, self.y = x_train, np.array(y_train_c)
            return self.x, self.y
        except StopIteration:
            print('Stop!')
            return None, None

    def __getitem__(self, idx):
        if not self.is_kickstarted:
            x, y = self.get_next_dataset()
            self.is_kickstarted = True
        else:
            x, y = self.x, self.y
        batch_x = x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        print('End of epoch==========================================', self.epoch, flush=True)
        # if self.is_checkpoint:
        #     self.is_checkpoint = False

        # FIXME #11: Here we save a checkpoint model if the number of epoch (which starts at 0, remember) is a multiple of *100*. Change this number as you like.
        if self.epoch % 100 == 0:
            # FIXME #12: Change this to False if you don't want to train the RFC.
            use_rlf = True
            if use_rlf:
                rlf, rfacc = utils_learning_rnn.compute_rlf_on_rnn(self.model, self.x, self.x_test, self.y_train, self.y_test, self.best_params_rf, config_learning.noofparallelthreads, use_graph=True, graph=self.graph)
            else:
                rlf = None

            # FIXME #13: Change this to the path you want your models to be saved to.
            save_to = os.path.join(config_learning.learnmodelspath, 'checkpoints_' + self.random_str)
            self.save_model(save_to, 'epoch'+str(self.epoch), rlf=rlf)

        # this_train_dir = os.path.join(self.trainpath, str(self.epoch+1))
        # if not os.path.exists(this_train_dir):
        #    if self.make_next_dataset() < 0:
        #        self.is_checkpoint = True
        # self.real_epoch += 1
        # self.epoch = self.real_epoch // 5

        self.epoch += 1
        self.get_next_dataset()
        if self.x is None:
            self.model.stop_training = True

