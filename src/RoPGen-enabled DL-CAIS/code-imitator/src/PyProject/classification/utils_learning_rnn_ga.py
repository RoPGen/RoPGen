import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Layer, Lambda
from tensorflow.keras.layers import LSTM, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
import tensorflow.keras.optimizers
from sklearn.model_selection import ParameterGrid
from tensorflow.keras import regularizers
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestClassifier
import sklearn.base
import tensorflow.keras.callbacks
import math

from ConfigurationLearning.ConfigurationLearningRNN import ConfigurationLearningRNN

# Crop the LSTM kernels and bias of the fullnet, for use in subnet. Since the LSTM kernels and bias are four parts concatenated together, we need to handle
# the parts separately and then put them back together.
# Params: W - kernel; U - recurrent kernel; b - bias
def crop_lstm_weights(W, U, b, fullnet_nounits, subnet_nounits, input_dim=None):
    W_i = W[:, :subnet_nounits] if input_dim is None else W[:input_dim, :subnet_nounits]
    W_f = W[:, fullnet_nounits: fullnet_nounits+subnet_nounits] if input_dim is None else W[:input_dim, fullnet_nounits: fullnet_nounits+subnet_nounits]
    W_c = W[:, fullnet_nounits * 2: fullnet_nounits * 2+subnet_nounits] if input_dim is None else W[:input_dim, fullnet_nounits * 2: fullnet_nounits * 2+subnet_nounits]
    W_o = W[:, fullnet_nounits * 3: fullnet_nounits * 3+subnet_nounits] if input_dim is None else W[:input_dim, fullnet_nounits * 3: fullnet_nounits * 3+subnet_nounits]
    W_new = np.hstack((W_i, W_f, W_c, W_o))

    U_i = U[:subnet_nounits, :subnet_nounits]
    U_f = U[:subnet_nounits, fullnet_nounits: fullnet_nounits+subnet_nounits]
    U_c = U[:subnet_nounits, fullnet_nounits * 2: fullnet_nounits * 2+subnet_nounits]
    U_o = U[:subnet_nounits, fullnet_nounits * 3: fullnet_nounits * 3+subnet_nounits]
    U_new = np.hstack((U_i, U_f, U_c, U_o))

    b_i = b[:subnet_nounits]
    b_f = b[fullnet_nounits: fullnet_nounits+subnet_nounits]
    b_c = b[fullnet_nounits * 2: fullnet_nounits * 2+subnet_nounits]
    b_o = b[fullnet_nounits * 3: fullnet_nounits * 3+subnet_nounits]
    b_new = np.hstack((b_i, b_f, b_c, b_o))

    #print(W_new.shape, U_new.shape, b_new.shape)
    return (W_new, U_new, b_new)

def my_model(input_dim_eq, output_dim_eq, width_mult=1.0, fullnet_model=None, denseneurons=128, lstmlayersno=1, denselayersno=3, optimizer=RMSprop(), nounits=32,
             dropout=0, l2reg=0, verbose=True):
    activation = "relu"
    input_shape_lstm = (1, input_dim_eq)
    lstmlayersno -= 1
    #assert denselayersno>=2
    denselayersno -= 2

    model = Sequential()

    return_seq_first = False if lstmlayersno == 0 else True
    #model.add(Lambda(lambda x: x, input_shape=input_shape_lstm))

    if fullnet_model is None:
        # Normal training, without GradAug.
        model.add(LSTM(nounits, name="LSTM_0", return_sequences=return_seq_first,
                       input_shape=input_shape_lstm, dropout=dropout))
        if lstmlayersno >= 1:
            for l in range(lstmlayersno):
                return_seq_last = False if l == (lstmlayersno - 1) else True
                model.add(LSTM(nounits, name="LSTM_"+str(l+1), dropout=dropout, return_sequences=return_seq_last))

        if denselayersno >= 1:
            for l in range(denselayersno):
                model.add(Dense(denseneurons, name="Dense_"+str(l+1), kernel_regularizer=regularizers.l2(l2reg)))
                model.add(Activation(activation))

        model.add(Dense(round(denseneurons*0.8), name="deep_representation", kernel_regularizer=regularizers.l2(l2reg)))

        model.add(Activation(activation))
        model.add(Dense(output_dim_eq, name="before_softmax", kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Activation('softmax'))
    else:
        # GradAug
        fullnet_nounits = nounits
        nounits = math.floor(nounits * width_mult)
        print('nounits: ', nounits)
        denseneurons = math.floor(denseneurons * width_mult)
        fullnet_weights = fullnet_model.get_layer(name="LSTM_0").get_weights()
        # Crop the fullnet weights to get subnet weights.
        kernel, recurrent_kernel, bias = crop_lstm_weights(fullnet_weights[0], fullnet_weights[1], fullnet_weights[2], fullnet_nounits, nounits)

        kernel_initializer = tf.keras.initializers.constant(kernel)
        recurrent_initializer = tf.keras.initializers.constant(recurrent_kernel)
        bias_initializer = tf.keras.initializers.constant(bias)
        # Add layer and initialize subnet weights.
        model.add(LSTM(nounits, name="LSTM_0", dropout=dropout, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=False, return_sequences=return_seq_first,
                       input_shape=input_shape_lstm))
        if lstmlayersno >= 1:
            for l in range(lstmlayersno):
                return_seq_last = False if l == (lstmlayersno - 1) else True
                fullnet_weights = fullnet_model.get_layer(name="LSTM_"+str(l+1)).get_weights()
                kernel, recurrent_kernel, bias = crop_lstm_weights(fullnet_weights[0], fullnet_weights[1], fullnet_weights[2], fullnet_nounits, nounits, input_dim=nounits)

                kernel_initializer = tf.keras.initializers.constant(kernel)
                recurrent_initializer = tf.keras.initializers.constant(recurrent_kernel)
                bias_initializer = tf.keras.initializers.constant(bias)
                model.add(LSTM(nounits, dropout=dropout, name="LSTM_"+str(l+1), kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=False, return_sequences=return_seq_last))

        if denselayersno >= 1:
            for l in range(denselayersno):
                fullnet_weights = fullnet_model.get_layer(name="Dense_"+str(l+1)).get_weights()
                subnet_weights = []
                subnet_weights.append(fullnet_weights[0][:nounits, :denseneurons])
                subnet_weights.append(fullnet_weights[1][:denseneurons])
                kernel_initializer = tf.keras.initializers.constant(subnet_weights[0])
                bias_initializer = tf.keras.initializers.constant(subnet_weights[1])
                model.add(Dense(denseneurons, name="Dense_"+str(l+1), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=regularizers.l2(l2reg)))
                model.add(Activation(activation))

        fullnet_weights = fullnet_model.get_layer(name="deep_representation").get_weights()
        subnet_weights = []
        subnet_weights.append(fullnet_weights[0][:denseneurons, :round(denseneurons*0.8)])
        subnet_weights.append(fullnet_weights[1][:round(denseneurons*0.8)])
        kernel_initializer = tf.keras.initializers.constant(subnet_weights[0])
        bias_initializer = tf.keras.initializers.constant(subnet_weights[1])
        model.add(Dense(round(denseneurons*0.8), name="deep_representation", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=regularizers.l2(l2reg)))

        model.add(Activation(activation))
        
        fullnet_weights = fullnet_model.get_layer(name="before_softmax").get_weights()
        subnet_weights = []
        subnet_weights.append(fullnet_weights[0][:round(denseneurons*0.8), :output_dim_eq])
        subnet_weights.append(fullnet_weights[1][:output_dim_eq])
        kernel_initializer = tf.keras.initializers.constant(subnet_weights[0])
        bias_initializer = tf.keras.initializers.constant(subnet_weights[1])
        model.add(Dense(output_dim_eq, name="before_softmax", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=regularizers.l2(l2reg)))
        model.add(Activation('softmax'))

    model.compile(loss=['categorical_crossentropy'], optimizer=optimizer,
                  metrics=['accuracy', 'categorical_accuracy'])
    if verbose is True:
        print(model.summary())

    return model


def split_params_into_rnn_rf(params: dict):
    params_rnn = {}
    params_rlf = {}
    for k, v in params.items():
        if k.startswith("RF_"):
            params_rlf[k[3:]] = v
        else:
            assert(k.startswith("RNN_"))
            params_rnn[k[4:]] = v
    return params_rnn, params_rlf



def compute_rlf_on_rnn(clf_best, x_train, x_test, y_train, y_test, params, noofparallelthreads, use_graph=False, graph=None):
    # D. Learn RF
    if not use_graph:
        get_deep_features = Model(clf_best.inputs, clf_best.layers[-3].output)
        x_train_deep = get_deep_features.predict([x_train])
        x_test_deep = get_deep_features.predict([x_test])
    else:
        with graph.as_default():
            get_deep_features = Model(clf_best.inputs, clf_best.layers[-3].output)
            x_train_deep = get_deep_features([x_train], training=False)
            x_test_deep = get_deep_features([x_test], training=False)


    rlf_deep = RandomForestClassifier(random_state=41, n_jobs=noofparallelthreads)
    rlf_deep.set_params(**params)

    rlf_deep.fit(x_train_deep, y_train)
    ypred = rlf_deep.predict(x_test_deep)
    cmps: np.ndarray = (y_test == ypred)
    rfaccuracy: np.float64 = np.sum(cmps) / np.shape(ypred)[0]

    return rlf_deep, rfaccuracy



def customized_grid_search_rnn(param_grid: dict, clf, listoftraintestsplits: list, cv_use_rnn_output: bool, noofparallelthreads):

    accuracy_per_param: list = []
    params_list: list = [params for params in ParameterGrid(param_grid)]

    if len(params_list) == 1:
        print("Only one param combi. No need for grid search!")
        return params_list[0], None

    for params_ in params_list:
        cv_results = []
        cv_times = []
        #print(params)

        params_rnn, params_rf = split_params_into_rnn_rf(params=params_)

        for x_train_cv, x_test_cv, y_train_cv, y_test_cv, y_train_c_cv, y_test_c_cv in listoftraintestsplits:
            #print(x_train_cv.shape)

            clf_copy = sklearn.base.clone(clf)
            clf_copy.set_params(**params_rnn)

            input_dim_eq = x_train_cv.shape[2]
            output_dim_eq = y_train_c_cv.shape[1]
            ad = keras.optimizers.Adam(lr=10e-4)
            early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=20, verbose=1, min_delta=0.0)
            dimsparams = dict(input_dim_eq = input_dim_eq, output_dim_eq = output_dim_eq,
                              callbacks=[early_stop], optimizer=ad)
            clf_copy.set_params(**dimsparams)

            time1 = time.time()
            clf_copy.fit(x_train_cv, y_train_c_cv)

            if cv_use_rnn_output is True:
               ypred_cv = clf_copy.predict(x_test_cv)
               theaccuracy: np.float64 = np.mean(np.argmax(y_test_c_cv, axis=1) == ypred_cv)
            else:
                _, theaccuracy = compute_rlf_on_rnn(clf_best=clf_copy, x_train=x_train_cv, x_test=x_test_cv,
                                                          y_train=y_train_cv, y_test=y_test_cv, params=params_rf,
                                                    noofparallelthreads=noofparallelthreads)

            time2 = time.time()
            cv_times.append( (time2-time1)*1000.0 )

            cv_results.append(theaccuracy)
            del clf_copy
        print("{1} (mean:{2}) by {0} (time: {3:.3f} ms)".format(params_, cv_results, np.mean(np.array(cv_results)),
                                                             np.mean(np.array(cv_times))))
        accuracy_per_param.append((np.mean(np.array(cv_results)), np.sqrt(np.var(np.array(cv_results)))))

    best_param_index: int = np.argmax(np.array([x[0] for x in accuracy_per_param]))
    print("Best param:", params_list[best_param_index], " with ", accuracy_per_param[best_param_index])

    return params_list[best_param_index], accuracy_per_param[best_param_index]
