import sys
import os

# FIXME #3: Change this to your project root.
repo_path = '/home/hx/data/code-imitator/'
sys.path.insert(1, os.path.join(repo_path, 'src', 'PyProject'))

from featureextractionV2.StyloFeaturesProxy import StyloFeaturesProxy
from featureextractionV2.StyloFeatures import StyloFeatures
from featureextractionV2.StyloUnigramFeatures import StyloUnigramFeatures

import math
import numpy as np
import pickle

from ConfigurationLearning.ConfigurationLearningRNN import ConfigurationLearningRNN
import ConfigurationGlobalLearning as Config
from classification.NovelAPI.Learning import Learning
from sklearn.model_selection import ParameterGrid

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.optimizers
import tensorflow.keras.callbacks
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical

import random
import string
import gc

# Given complete training set feature matrices, return a part of them depending on what 'dataset_to_return' is.
# For example, your training set may consist of the original training samples and the MCTS adversarial samples,
# and you can modify this function to return only one of them.
def get_matrices(x_subnet_train, y_subnet_train, doclabels_subnet_train, authors_subnet_train, dataset_to_return='aug3'):
    i = 0
    # Just ignore the 'fullnet' in these names... Wrote this function some while ago and the names are inaccurate now.
    fullnet_x_train = None
    fullnet_y_train = None
    fullnet_doclabels = None
    fullnet_authors = None

    # Iterate over the filenames of samples in the training set.
    for doclabel in doclabels_subnet_train:
        # FIXME #1: Tailor these cases to your needs. Here, 'doclabel' is the filename of a sample in the training set,
        # and we determine if it is what we want to return by its filename. For example, you may want to name all your
        # MCTS adversarial samples with the string 'mcts_unt' in it, so that they can be filtered out when the original
        # samples are requested.
        # Note: if you're using our adversarial GitHub C, GitHub Java, or GCJ Java dataset,
        # you need to uncomment the corresponding part below.
        if dataset_to_return == 'aug3':
            if 'mctsunt' in doclabel or 'mcts_unt' in doclabel or 'mcts_t' in doclabel:
                i += 1
                continue
            '''
            # Uncomment for GitHub Java and GCJ Java
            if 'transform' in doclabel:
                i += 1
                continue
            '''
        '''
        # Uncomment for GitHub C
        elif dataset_to_return == 'aug2':
            if '##' in doclabel:
                i += 1
                continue
        '''
        '''
        # Uncomment for GitHub Java and GCJ Java
        elif dataset_to_return == 'aug2':
            if '##' in doclabel and 'transform' not in doclabel:
                i += 1
                continue
        '''
        elif dataset_to_return == 'mctstemp':
            if 'mctsunt' not in doclabel and 'mctst' not in doclabel:
                i += 1
                continue
        elif dataset_to_return == 'mctsadv':
            if len(doclabel.split('_')) > 3 and 'mcts_unt' not in doclabel and 'mcts_t' not in doclabel:
                i += 1
                continue
        elif dataset_to_return == 'mctsadvt':
            if 'mcts_t' not in doclabel:
                i += 1
                continue
        elif dataset_to_return == 'mctsadvunt':
            if 'mcts_unt' not in doclabel:
                i += 1
                continue
        elif dataset_to_return == 'orig':
            if len(doclabel.split('_')) > 3:
                i += 1
                continue

        if fullnet_x_train is None:
            fullnet_x_train = x_subnet_train[i, :, :]
            fullnet_y_train = y_subnet_train[i]
            fullnet_doclabels = doclabels_subnet_train[i]
            fullnet_authors = authors_subnet_train[i]
        else:
            fullnet_x_train = np.vstack((fullnet_x_train, x_subnet_train[i, :]))
            fullnet_y_train = np.append(fullnet_y_train, y_subnet_train[i])
            fullnet_doclabels = np.append(fullnet_doclabels, doclabels_subnet_train[i])
            fullnet_authors = np.append(fullnet_authors, authors_subnet_train[i])
        i += 1
    fullnet_x_train = fullnet_x_train.reshape(fullnet_x_train.shape[0], 1, fullnet_x_train.shape[1])
    return fullnet_x_train, fullnet_y_train, fullnet_doclabels, fullnet_authors

def train_step(fullnet_batch_num, x, y, doclabels_batch, width_mult_list):
    train_vars = model.trainable_variables
    # Create empty gradient list (not a tf.Variable list)
    accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    with tf.GradientTape() as tape:
        # fullnet - forward pass in training mode
        max_predictions = model(x, training=True)
        # fullnet - calculate batch loss function
        loss = tf.keras.losses.CategoricalCrossentropy()(y, max_predictions)
    print('fullnet batch loss {}'.format(loss))
    # fullnet - calculate the gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # Count how many LSTM layer variables we have.
    lstm_var_cnt = 0
    for model_var in model.trainable_variables:
        print(model_var.name)
        if 'LSTM' in model_var.name: lstm_var_cnt += 1
    accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads)]
    sub_loss = 0

    # Below is subnet.
    if init_min_width > 0:
        for idx, width_mult in enumerate(sorted(width_mult_list, reverse=True)):
            print(width_mult)
            # subnet - create a subnet model, with width being 'width_mult'.
            subnet_model = utils_learning_rnn.my_model(**clf_best.filter_sk_params(utils_learning_rnn.my_model), width_mult=width_mult, fullnet_model=model)
            subnet_model.summary()

            if idx == 0:
                if fullnet_batch_num > len(subnet_train_dataset)-1: continue
                subnet_batch = subnet_train_dataset[fullnet_batch_num]
            elif idx == 1:
                if fullnet_batch_num > len(subnet2_train_dataset)-1: continue
                subnet_batch = subnet2_train_dataset[fullnet_batch_num]
            elif idx == 2:
                if fullnet_batch_num > len(subnet3_train_dataset)-1: continue
                subnet_batch = subnet3_train_dataset[fullnet_batch_num]
            #elif idx == 3:
            #    if fullnet_batch_num > len(subnet4_train_dataset)-1: continue
            #    subnet_batch = subnet4_train_dataset[fullnet_batch_num]

            subnet_features, subnet_labels, subnet_doclabels, subnet_authors = subnet_batch[0], subnet_batch[1], subnet_batch[2], subnet_batch[3]

            print("Start of subnet batch {}".format(fullnet_batch_num))
            labels_c = tf.convert_to_tensor(to_categorical(subnet_labels, num_classes=num_classes))

            # subnet - forward pass and loss calculation
            with tf.GradientTape() as tape:
                predictions = subnet_model(subnet_features, training=True)
                sub_loss = tf.keras.losses.CategoricalCrossentropy()(labels_c, predictions)
            
            predictions_c = np.argmax(predictions, axis=1)
            y_c = np.argmax(labels_c, axis=1)
            eq = (predictions_c == y_c)
            print('subnet acc: ', np.count_nonzero(eq) / len(eq))
            print('subnet batch loss {}'.format(sub_loss))

            # subnet - calculate subnet gradients.
            grads = tape.gradient(sub_loss, subnet_model.trainable_variables)

            # Now we're going to pad the subnet gradients, since subnet and fullnet gradients have different shapes and we must do some padding
            # before we can accumulate gradients from sub- and full-net.
            # LSTM gradients padding, each grad matrix is divided into 4 submatrices, corresponding to input, forget, cell, and output gate
            for i in range(lstm_var_cnt):
                accum_gradient_np = accum_gradient[i].numpy()
                grads_np = grads[i].numpy()
                if grads_np.ndim == 1:
                    grads_np_i = grads_np[:grads_np.shape[0] // 4]
                    grads_np_f = grads_np[grads_np.shape[0] // 4: grads_np.shape[0] // 4 * 2]
                    grads_np_c = grads_np[grads_np.shape[0] // 4 * 2: grads_np.shape[0] // 4 * 3]
                    grads_np_o = grads_np[grads_np.shape[0] // 4 * 3:]
                    grads_np_i = np.pad(grads_np_i, (0, accum_gradient_np.shape[0]//4-grads_np.shape[0]//4), mode='constant')
                    grads_np_f = np.pad(grads_np_f, (0, accum_gradient_np.shape[0]//4-grads_np.shape[0]//4), mode='constant')
                    grads_np_c = np.pad(grads_np_c, (0, accum_gradient_np.shape[0]//4-grads_np.shape[0]//4), mode='constant')
                    grads_np_o = np.pad(grads_np_o, (0, accum_gradient_np.shape[0]//4-grads_np.shape[0]//4), mode='constant')
                    grads_np = np.hstack((grads_np_i, grads_np_f, grads_np_c, grads_np_o))
                elif grads_np.ndim == 2:
                    grads_np_i = grads_np[:, :grads_np.shape[1] // 4]
                    grads_np_f = grads_np[:, grads_np.shape[1] // 4: grads_np.shape[1] // 4 * 2]
                    grads_np_c = grads_np[:, grads_np.shape[1] // 4 * 2: grads_np.shape[1] // 4 * 3]
                    grads_np_o = grads_np[:, grads_np.shape[1] // 4 * 3:]
                    grads_np_i = np.pad(grads_np_i, ((0, accum_gradient_np.shape[0]-grads_np.shape[0]), (0, accum_gradient_np.shape[1]//4-grads_np.shape[1]//4)), mode='constant')
                    grads_np_f = np.pad(grads_np_f, ((0, accum_gradient_np.shape[0]-grads_np.shape[0]), (0, accum_gradient_np.shape[1]//4-grads_np.shape[1]//4)), mode='constant')
                    grads_np_c = np.pad(grads_np_c, ((0, accum_gradient_np.shape[0]-grads_np.shape[0]), (0, accum_gradient_np.shape[1]//4-grads_np.shape[1]//4)), mode='constant')
                    grads_np_o = np.pad(grads_np_o, ((0, accum_gradient_np.shape[0]-grads_np.shape[0]), (0, accum_gradient_np.shape[1]//4-grads_np.shape[1]//4)), mode='constant')
                    grads_np = np.hstack((grads_np_i, grads_np_f, grads_np_c, grads_np_o))
                grads[i] = tf.convert_to_tensor(grads_np)

            # For non-LSTM layers this will be much easier: just pad the matrix as a whole with zeros.
            for i in range(lstm_var_cnt, len(grads)):
                accum_gradient_np = accum_gradient[i].numpy()
                grads_np = grads[i].numpy()
                if grads_np.ndim == 1:
                    grads_np = np.pad(grads_np, (0, accum_gradient_np.shape[0]-grads_np.shape[0]), mode='constant')
                elif grads_np.ndim == 2:
                    grads_np = np.pad(grads_np, ((0, accum_gradient_np.shape[0]-grads_np.shape[0]), (0, accum_gradient_np.shape[1]-grads_np.shape[1])), mode='constant')
                grads[i] = tf.convert_to_tensor(grads_np)

            # Accumulate the gradients.
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads)]
            # We were having memory issues with TF 1.x, but these lines are pretty much useless now since we upgraded to TF 2.x.
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            del subnet_model
            gc.collect()

    # backpropagate
    optimizer.apply_gradients(zip(accum_gradient, model.trainable_variables))

    return loss, sub_loss

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
        return kerasmodelpath

def save_model(model, trainfiles, sc, rlf, modelsavedir, curproblem):
    from classification.LearnSetups.LearnSetupRNNRF import LearnSetupRNNRF
    testlearnsetup = LearnSetupRNNRF(data_final_train=trainfiles, data_final_test=None,
                                         clf=None, rlf=rlf, stdscaler=sc)
    if not os.path.exists(modelsavedir):
        os.makedirs(modelsavedir)
    save_learnsetup(modelsavedir=modelsavedir, curproblem=curproblem, learn_method='RNN',
                          threshold=threshold, testlearnsetup=testlearnsetup)
    return save_keras_model(modelsavedir=modelsavedir, curproblem=curproblem, learn_method='RNN',
                           threshold=threshold, keras_model=model, keras_hist=None)


if __name__ == '__main__':
    # FIXME #2: Select the GPU you want to use. Set to -1 to disable GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    np.set_printoptions(threshold=sys.maxsize)

    # Further parameters:
    # we use the following dataset.
    # FIXME #4: Change this to your training set.
    datasetpath = os.path.join(repo_path, "data", "robust_model_training_set")
    # FIXME #5: Change this to your testing set (could be arbitrary, doesn't affect training results). Technically, the testing set is of no use
    # throughout this script, but the method that trains the RF classifier (i.e., compute_rlf_on_rnn(), which was written by the MCTS attack authors)
    # requires a testing set in order to compute the accuracy, so better make it happy.
    testpath = os.path.join(repo_path, "data", "testset")
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

    # Extract features.
    # FIXME #6: Change 'noprogrammers' below to the number of authors in your training set.
    unigrammmatrix_train = StyloUnigramFeatures(inputdata=datasetpath,
                                                nocodesperprogrammer=8,
                                                noprogrammers=204,
                                                binary=False, tf=True, idf=True,
                                                ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                                trainobject=None)
    trainfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=unigrammmatrix_train)
    # FIXME #7: Change 'noprogrammers' below to the number of authors in your testing set.
    unigrammmatrix_test = StyloUnigramFeatures(inputdata=testpath,
                                                nocodesperprogrammer=1,
                                                noprogrammers=1,
                                                binary=False, tf=True, idf=True,
                                                ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                                trainobject=unigrammmatrix_train)
    testfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=unigrammmatrix_test)

    learning: Learning = Learning()
    threshold = 800

    import classification.utils_learning_rnn_ga as utils_learning_rnn
    feature_dim = threshold
    # FIXME #8: Change 'num_classes' to the number of authors in your training set, and 'batch_size' to whatever you want.
    num_classes = 204
    batch_size = 128

    if config_learning.hyperparameters is None:
        # FIXME #9: Change 'RNN_epochs' to the max number of training epochs. Note that the epoch number starts at 0.
        param_grid = {
                    "RNN_epochs": [601],
                    "RNN_nounits": [288],
                    "RNN_dropout": [0.6],
                    "RNN_lstmlayersno": [3],
                    "RNN_denselayersno": [3],
                    "RNN_l2reg": [0.00001],
                    "RNN_denseneurons": [round(0.45*feature_dim)]
                    }
    else:
        param_grid = config_learning.hyperparameters
        param_grid['RNN_denseneurons'] = [round(x * feature_dim) for x in param_grid['RNN_denseneurons']]

    if config_learning.cv_optimize_rlf_params:
        param_grid_rf = {"RF_n_estimators": [250],
                    "RF_max_features": [0.3, 0.6, 'auto'],
                    "RF_max_depth": [10, 25, 50, 75, None],
                    "RF_min_samples_leaf": [6, 12, 1],
                    }
        param_grid.update(param_grid_rf)
    kerasclf = KerasClassifier(build_fn=utils_learning_rnn.my_model, batch_size=batch_size, input_dim_eq=feature_dim, output_dim_eq=num_classes,
                        optimizer="Adam", verbose=0)
    
    # FIXME #10: This was a grid search in the original version of code. For time reasons, we got rid of that and manually set the params,
    # so if you want the grid search back, you'll have to change this.
    best_params_, best_params_acc = [params for params in ParameterGrid(param_grid)][0], None
    best_params_rnn, best_params_rf = utils_learning_rnn.split_params_into_rnn_rf(params=best_params_)

    # Learn on best params
    optimizer = tensorflow.keras.optimizers.Adam(lr=10e-4)
    clf_best = KerasClassifier(build_fn=utils_learning_rnn.my_model, batch_size=batch_size,
                            input_dim_eq=feature_dim, output_dim_eq=num_classes,
                            optimizer=optimizer,
                            callbacks=[],
                            verbose=1)
    clf_best.set_params(**best_params_rnn)

    model = utils_learning_rnn.my_model(**clf_best.filter_sk_params(utils_learning_rnn.my_model))
    model.summary()

    print(">Whole train set: TRAIN:", trainfiles.getfeaturematrix().shape, "TEST:", testfiles.getfeaturematrix().shape)
    train_obj, test_obj = learning._tfidf_feature_selection(train_obj=trainfiles, test_obj=testfiles,
                                                    config_learning=config_learning,
                                                    threshold=threshold)

    # Preparing the fullnet and subnet training feature matrices, the labels, etc.
    x_subnet_train_, y_subnet_train_ = train_obj.getfeaturematrix(), train_obj.getlabels()
    y_subnet_train_c = to_categorical(y_subnet_train_, num_classes=num_classes)
    doclabels_subnet_train_, authors_subnet_train_ = train_obj.getdoclabels(), train_obj.getauthors()
    x_test_, y_test = test_obj.getfeaturematrix(), test_obj.getlabels()

    from sklearn import preprocessing
    sc = preprocessing.StandardScaler().fit(x_subnet_train_.todense())
    x_subnet_train_ = sc.transform(x_subnet_train_.todense())
    x_test_ = sc.transform(x_test_.todense())
    x_subnet_train_ = x_subnet_train_.reshape(x_subnet_train_.shape[0], 1, x_subnet_train_.shape[1])
    x_test_ = x_test_.reshape(x_test_.shape[0], 1, x_test_.shape[1])

    x_train, y_train, doclabels, authors = get_matrices(x_subnet_train_, y_subnet_train_, doclabels_subnet_train_, authors_subnet_train_)
    # FIXME #14: Comment out the following three lines of code AND uncomment the subsequent line below if you're using our adversarial dataset other than GCJ C++.
    # If you're NOT using our dataset, change the argument 'dataset_to_return' as needed to change the input data to the subnets.
    x_subnet_train, y_subnet_train, doclabels_subnet_train, authors_subnet_train = get_matrices(x_subnet_train_, y_subnet_train_, doclabels_subnet_train_, authors_subnet_train_, dataset_to_return='mctsadvt')
    x_subnet2_train, y_subnet2_train, doclabels_subnet2_train, authors_subnet2_train = get_matrices(x_subnet_train_, y_subnet_train_, doclabels_subnet_train_, authors_subnet_train_, dataset_to_return='mctstemp')
    x_subnet3_train, y_subnet3_train, doclabels_subnet3_train, authors_subnet3_train = get_matrices(x_subnet_train_, y_subnet_train_, doclabels_subnet_train_, authors_subnet_train_, dataset_to_return='mctsadvunt')
    # Uncomment for GitHub C, GitHub Java, or GCJ Java
    # x_subnet_train, y_subnet_train, doclabels_subnet_train, authors_subnet_train = get_matrices(x_subnet_train_, y_subnet_train_, doclabels_subnet_train_, authors_subnet_train_, dataset_to_return='aug2')

    # Split the training set into batches.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, doclabels, authors))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # We want to make sure the fullnet and the subnet have the same number of batches, so here we calculate the batch size for the subnet.
    subnet_batchsize = math.ceil(x_subnet_train.shape[0] / len(train_dataset))
    print('fullnet batch num: ', len(train_dataset))
    print('subnet batch size: ', subnet_batchsize)
    subnet_train_dataset = tf.data.Dataset.from_tensor_slices((x_subnet_train, y_subnet_train, doclabels_subnet_train, authors_subnet_train))
    subnet_train_dataset = list(subnet_train_dataset.shuffle(buffer_size=1024).batch(subnet_batchsize).as_numpy_iterator())

    subnet2_batchsize = math.ceil(x_subnet2_train.shape[0] / len(train_dataset))
    subnet2_train_dataset = tf.data.Dataset.from_tensor_slices((x_subnet2_train, y_subnet2_train, doclabels_subnet2_train, authors_subnet2_train))
    subnet2_train_dataset = list(subnet2_train_dataset.shuffle(buffer_size=1024).batch(subnet2_batchsize).as_numpy_iterator())

    subnet3_batchsize = math.ceil(x_subnet3_train.shape[0] / len(train_dataset))
    subnet3_train_dataset = tf.data.Dataset.from_tensor_slices((x_subnet3_train, y_subnet3_train, doclabels_subnet3_train, authors_subnet3_train))
    subnet3_train_dataset = list(subnet3_train_dataset.shuffle(buffer_size=1024).batch(subnet3_batchsize).as_numpy_iterator())

    init_min_width = 0.8
    init_max_width = 1.0

    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 5))
    # FIXME #11: Change this to the path you want your models to be saved to.
    save_to = os.path.join(config_learning.learnmodelspath, 'gradaug_checkpoints_' + random_str)

    for epoch in range(param_grid['RNN_epochs'][0]):
        batch_losses = []
        print('Start of epoch {}'.format(epoch))
        # enumerate the training set in batches
        for (batch, (features, labels, doclabels_batch, authors)) in enumerate(train_dataset):
            print("Start of batch {}".format(batch))
            train_loss = 0
            num_subnet = 3
            if init_min_width > 0:
                # Randomly sample subnet width.
                min_width = init_min_width
                max_width = init_max_width
                width_mult_list = [min_width]
                sampled_width = list(np.random.uniform(min_width, max_width, num_subnet-1))
                width_mult_list.extend(sampled_width)

                labels_c = to_categorical(labels, num_classes=num_classes)
                # Train on this batch.
                train_full_loss, sub_loss = train_step(batch, features, labels_c, doclabels_batch, width_mult_list)
            
            print('Training loss for batch {} is {}'.format(batch, train_full_loss))
            batch_losses.append(train_full_loss)

        print('Training loss for epoch {} is {}'.format(epoch, sum(batch_losses) / len(batch_losses)))
        with open('losses_gradaug.txt', 'a') as f:
            f.write('Epoch ' + str(epoch) + ', loss: ' + str(sum(batch_losses) / len(batch_losses)) + '\n')

        # Train the RF classifier.
        # FIXME #12: Here we save a checkpoint model if the number of epoch (which starts at 0, remember) is a multiple of *100*. Change this number as you like.
        if epoch % 500 == 0:
            # FIXME #13: Change this to False if you don't want to train the RFC.
            use_rlf = True
            if use_rlf:
                rlf, rfacc = utils_learning_rnn.compute_rlf_on_rnn(model, x_subnet_train_, x_test_, y_subnet_train_, y_test, best_params_rf, config_learning.noofparallelthreads)
            else:
                rlf = None
            save_model(model, train_obj, sc, rlf, save_to, 'epoch'+str(epoch))