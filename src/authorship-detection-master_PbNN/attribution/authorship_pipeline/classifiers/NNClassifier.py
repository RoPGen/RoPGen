import time
from typing import Tuple, List, Union, Dict, Counter
import pickle
import numpy as np
import pandas as pd
import torch,os,shutil
from torch import optim, nn
from torch.utils.data import DataLoader
import classifiers.GeneMergeCsv as GeneMergeCsv
from util import ProcessedFolder, ProcessedSnapshotFolder

import transform_files

from classifiers.BaseClassifier import BaseClassifier, ClassificationResult, compute_classification_result
from classifiers.config import Config
from model.ProjectClassifier import ProjectClassifier
from preprocessing.context_split import ContextSplit
from util import ProcessedFolder

import random

from shutil import copy
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

class NNClassifier(BaseClassifier):
    def __init__(self, config: Config, project_folder: ProcessedFolder, change_entities: pd.Series,
                 change_to_time_bucket: Dict, min_max_count: Tuple[int, int], author_occurrences: Counter,
                 context_splits: List[ContextSplit]):
        super(NNClassifier, self).__init__(config, project_folder, change_entities, change_to_time_bucket,
                                           min_max_count, author_occurrences, context_splits)

    def __sample_loaders(self, fold_ind: Union[int, Tuple[int, int]] = 0 ,is_epoch_attack=False,batch_size=-1) -> Tuple[DataLoader, DataLoader]:
        '''
        The function creates a train and test loader for the given fold.
        
        :param self: This is the object that is being called
        :param fold_ind: The index of the fold to use, defaults to 0
        :type fold_ind: Union[int, Tuple[int, int]] (optional)
        :param is_epoch_attack: If True, the attack is performed on the whole dataset, defaults to False
        (optional)
        :param batch_size: The batch size to use for training
        :return: The train_loader and test_loader.
        '''
        if batch_size == -1:
            batch_size=self.config.batch_size()
        train_dataset, test_dataset = self._split_train_test(self._loader, fold_ind, is_epoch_attack=is_epoch_attack ,pad=True)
        # Create training and validation dataset.
        train_loader=[]
        if self.config.final_test() == 0 and is_epoch_attack==False:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size)
        return train_loader, test_loader
        

    def repair_labels(self,labels,train_original_labels,cur_original_labels=None):
        '''
        The function is used to repair the labels of the test set.
            The labels of the test set are not in the same order as the labels of the train set.
            So we need to repair the labels of the test set.
        
        :param self: the instance of the class
        :param labels: the predicted labels
        :param train_original_labels: the original labels of the training set
        :param cur_original_labels: the original labels of the current batch
        :return: The labels are being returned as a tensor.
        '''
        labels_list = labels.numpy().tolist()
        
        # test
        original_labels = self._loader.original_labels()
        # train
        '''
        if all(cur_original_labels == None):
            original_labels = self._loader.original_labels()
        else:
            original_labels = cur_original_labels
        '''
        result = []
        for i in labels_list:
            result.append(train_original_labels.index(original_labels[i]))  
        result = torch.LongTensor(result)
        return result

    def __train(self, train_loader, test_loaders, model, optimizer, loss_function, n_epochs, log_batches, batch_size,
                fold_ind, should_train, train_original_labels):

        print("Start training")
        accuracies = [ClassificationResult(0, 0, 0, 0) for _ in range(len(test_loaders))]
        if not should_train:
            n_epochs = 1
        for epoch in range(n_epochs):
            print("Epoch #{}".format(epoch + 1))
            if should_train:
                model.train()
                current_loss = 0
                start_time = time.time()
                for n_batch, sample in enumerate(train_loader):
                    starts, paths, ends, labels = sample['starts'], sample['paths'], sample['ends'], sample['labels']
                    optimizer.zero_grad()
                    predictions = model((starts, paths, ends))
                    loss = loss_function(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    current_loss += loss.item()
                    if (n_batch + 1) % log_batches == 0:
                        print("After {} batches: average loss {}".format(n_batch + 1, current_loss / log_batches))
                        print(f"Throughput {int(log_batches * batch_size / (time.time() - start_time))} examples / sec")
                        current_loss = 0
                        start_time = time.time()
                model.eval()
            with torch.no_grad():
                for i, test_loader in enumerate(test_loaders):
                    total = len(test_loader.dataset)
                    predictions = np.zeros(total)
                    targets = np.zeros(total)
                    cur = 0
                    test_loss = 0.
                    n_batches = 0
                    for sample in test_loader:
                        starts, paths, ends, labels = sample['starts'], sample['paths'], sample['ends'], sample['labels']
                        #--------修复labels
                        if self.config.final_test() == 1:
                            labels = self.repair_labels(labels,train_original_labels.tolist())
                        batched_predictions = model((starts, paths, ends))
                        test_loss += loss_function(batched_predictions, labels)
                        n_batches += 1

                        batched_predictions = np.argmax(batched_predictions, axis=1) #np.argmax取预测概率最大值对应的索引
                        batched_targets = labels
                        predictions[cur:cur + len(batched_predictions)] = batched_predictions
                        targets[cur:cur + len(batched_targets)] = batched_targets
                        cur += len(batched_predictions)
                    classification_result = compute_classification_result(targets, predictions, fold_ind)
                    print(f"classification results: {classification_result}")
                    accuracies[i] = max(accuracies[i], classification_result, key=lambda cl: cl.accuracy)
        print("Training completed")
        return accuracies


    # Using GradAug method to train the model
    def __train_GradAug(self, train_loader, test_loaders, model, optimizer, loss_function, n_epochs, log_batches, batch_size,
                fold_ind, should_train, train_original_labels):
        '''
        1. Load the data
        2. Train the model
        3. Test the model
        4. Save the model
        5. Save the results
        
        :param self: the class itself
        :param train_loader: the training data loader
        :param test_loaders: a list of DataLoader objects, one for each test set
        :param model: the model to train
        :param optimizer: Adam
        :param loss_function: the loss function to use
        :param n_epochs: The number of epochs to train for
        :param log_batches: how often to print the loss
        :param batch_size: The number of samples in each batch
        :param fold_ind: the index of the fold to be trained
        :param should_train: whether to train the model or not
        :param train_original_labels: the labels of the training set
        :return: The accuracy of the model on the test set.
        '''
        print("Start training")
        accuracies = [ClassificationResult(0, 0, 0, 0) for _ in range(len(test_loaders))]
        if not should_train:
            n_epochs = 1
        change_entities = None
        author_occurrences = None
        change_to_time_bucket = None
        context_splits = None
        if should_train:
            print("Load subnet data")
            project_folder_tmp = ProcessedSnapshotFolder("/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/processed/grad/githubc2/train_aug2_mcts/c/")
            classifier_sub = NNClassifier(self.config, project_folder_tmp, change_entities, change_to_time_bucket,
                            self.config.min_max_count(), author_occurrences, context_splits)
            new_batch_size = int((classifier_sub._loader._labels.size-classifier_sub._n_classes)/(train_loader.dataset._size/train_loader.batch_size))
            print("new_batch_size %d"%new_batch_size)
            train_loader_sub, test_loaders_sub = classifier_sub.__sample_loaders(fold_ind,batch_size=new_batch_size)                                                           
            if type(test_loaders_sub) is DataLoader:
                test_loaders_sub = [test_loaders_sub]
        
        for epoch in range(n_epochs):        
            print("Epoch #{}".format(epoch + 1))        
            if should_train:
                model.train()
                current_loss = 0
                start_time = time.time()
                for n_batch, sample in enumerate(train_loader):
                    starts1, paths1, ends1, labels1 = sample['starts'], sample['paths'], sample['ends'], sample['labels']
                    optimizer.zero_grad()
                    min_width = 0.8
                    max_width = 1.0
                    num_subnet = 3
                    resos = [32, 28, 24]
                    criterion = nn.CrossEntropyLoss()
                    width_mult_list = [min_width]
                    sampled_width = list(np.random.uniform(min_width, max_width, num_subnet-1))   
                    width_mult_list.extend(sampled_width)
                    model.apply(lambda m: setattr(m, 'width_mult', max_width))
                    max_output = model((starts1, paths1, ends1))
                    print("Training full network")
                    loss = criterion(max_output, labels1)
                    print("full network's loss: "+str(loss.item()))
                    current_loss += loss.item()
                    loss.backward()                     
                    max_output_detach = max_output.detach()
      
                    if min_width > 0:  # randwidth
                        for width_mult in sorted(width_mult_list, reverse=True):                  
                            model.apply(
                                lambda m: setattr(m, 'width_mult', width_mult))
                            resolution = resos[random.randint(0, len(resos)-1)]                                
                            change_entities = None
                            author_occurrences = None
                            change_to_time_bucket = None
                            context_splits = None
                            for sub_n_batch, sample in enumerate(train_loader_sub):
                                if sub_n_batch != n_batch:
                                    continue
                                starts, paths, ends, labels = sample['starts'], sample['paths'], sample['ends'], sample['labels']
                                output = model((starts, paths, ends))
                                loss = criterion(output, labels)
                                print("sub network' loss: "+str(loss.item()))
                                current_loss += loss.item()
                                loss.backward()
                            with torch.no_grad():
                                for i, test_loaders_1 in enumerate(test_loaders_sub):
                                    total = len(test_loaders_1.dataset)
                                    predictions = np.zeros(total)
                                    targets = np.zeros(total)
                                    cur = 0
                                    test_loss = 0.
                                    n_batches = 0
                                    for sample in test_loaders_1:
                                        starts, paths, ends, labels = sample['starts'], sample['paths'], sample['ends'], sample['labels']
                                        batched_predictions = model((starts, paths, ends))
                                        n_batches += 1
                                        batched_predictions = np.argmax(batched_predictions, axis=1) 
                                        batched_targets = labels
                                        predictions[cur:cur + len(batched_predictions)] = batched_predictions
                                        targets[cur:cur + len(batched_targets)] = batched_targets
                                        cur += len(batched_predictions)
                                    classification_result = compute_classification_result(targets, predictions, fold_ind)
                                    print(f"sub_network's classification results: {classification_result}")
                                    accuracies = [ClassificationResult(0, 0, 0, 0) for _ in range(len(test_loaders_sub))]
                                    accuracies[i] = max(accuracies[i], classification_result, key=lambda cl: cl.accuracy)
                    optimizer.step()
                    if (n_batch + 1) % log_batches == 0:
                        start_time = time.time()
                model.eval()
            if type(test_loaders) is DataLoader:
                test_loaders = [test_loaders]
            
            with torch.no_grad():
                for i, test_loader in enumerate(test_loaders):
                    total = len(test_loader.dataset)
                    predictions = np.zeros(total)
                    targets = np.zeros(total)
                    cur = 0
                    test_loss = 0.
                    n_batches = 0
                    for sample in test_loader:
                        starts, paths, ends, labels = sample['starts'], sample['paths'], sample['ends'], sample['labels']
                        # repair labels                       
                        if self.config.final_test() == 1:
                            labels = self.repair_labels(labels,train_original_labels.tolist())
                        model.apply(
                                lambda m: setattr(m, 'width_mult', 1.0))
                        batched_predictions = model((starts, paths, ends))
                        test_loss += loss_function(batched_predictions, labels)
                        n_batches += 1

                        batched_predictions = np.argmax(batched_predictions, axis=1)
                        batched_targets = labels
                        predictions[cur:cur + len(batched_predictions)] = batched_predictions
                        targets[cur:cur + len(batched_targets)] = batched_targets
                        cur += len(batched_predictions)
                    classification_result = compute_classification_result(targets, predictions, fold_ind)
                    
                    print(f"full_network's classification results: {classification_result}")
                    accuracies[i] = max(accuracies[i], classification_result, key=lambda cl: cl.accuracy)
            current_loss = 0
        print("Training completed")
        return accuracies     


    def __run_classifier(self, train_loader: DataLoader, test_loaders: Union[DataLoader, List[DataLoader]], fold_ind,train_original_labels) \
            -> Union[float, List[float]]:
        if fold_ind not in self.models:
            model = ProjectClassifier(self._loader.tokens().size*10,
                                      self._loader.paths().size*10,
                                      dim=self.config.hidden_dim(),
                                      n_classes=self._loader.n_classes())
            should_train = True
        else:
            model = self.models[fold_ind]
            should_train = False
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate())
        loss_function = nn.CrossEntropyLoss()
        if type(test_loaders) is DataLoader:
            test_loaders = [test_loaders]
        training_method = self.config.training_method()
        if training_method == "original":
            accuracies = self.__train(train_loader, test_loaders, model, optimizer, loss_function,
                                    n_epochs=self.config.epochs(),
                                    log_batches=self.config.log_batches(),
                                    batch_size=self.config.batch_size(),
                                    fold_ind=fold_ind, should_train=should_train, train_original_labels=train_original_labels)
        elif training_method == "GradAug":
            accuracies = self.__train_GradAug(train_loader, test_loaders, model, optimizer, loss_function,
                                    n_epochs=self.config.epochs(),
                                    log_batches=self.config.log_batches(),
                                    batch_size=self.config.batch_size(),
                                    fold_ind=fold_ind, should_train=should_train, train_original_labels=train_original_labels)

        if fold_ind not in self.models:
            self.models[fold_ind] = model

        if len(test_loaders) == 1:
            return max(accuracies, key=lambda cl: cl.accuracy)
        else:
            return accuracies

    def run(self, fold_indices: Union[List[int], List[Tuple[int, int]]], train_original_labels) \
            -> Tuple[float, float, List[ClassificationResult]]:
        scores = []
        for fold_ind in fold_indices:
            train_loader, test_loader = self.__sample_loaders(fold_ind)
            scores.append(self.__run_classifier(train_loader, test_loader, fold_ind,train_original_labels))
        mean = float(np.mean([score.accuracy for score in scores]))
        std = float(np.std([score.accuracy for score in scores]))
        return mean, std, scores, self._loader.original_labels()
