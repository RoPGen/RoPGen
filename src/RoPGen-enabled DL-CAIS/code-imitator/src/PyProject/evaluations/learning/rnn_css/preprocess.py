
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np

def preprocess(num_classes, config_learning, x_train, x_test, y_train, y_test):
    # A. Preprocessing
    '''
    for i in range(len(listoftraintestsplits)):
        if config_learning.scale:
            sc = preprocessing.StandardScaler().fit(listoftraintestsplits[i][0].todense())
            listoftraintestsplits[i][0] = sc.transform(listoftraintestsplits[i][0].todense())
            listoftraintestsplits[i][1] = sc.transform(listoftraintestsplits[i][1].todense())
        else:
            listoftraintestsplits[i][0] = np.array(listoftraintestsplits[i][0].todense())
            listoftraintestsplits[i][1] = np.array(listoftraintestsplits[i][1].todense())

        listoftraintestsplits[i].append(to_categorical(listoftraintestsplits[i][2], num_classes=num_classes))
        listoftraintestsplits[i].append(to_categorical(listoftraintestsplits[i][3], num_classes=num_classes))

        # listoftraintestsplits[i][0], listoftraintestsplits[i][2] = shuffle(listoftraintestsplits[i][0],
        #                                                                    listoftraintestsplits[i][2],
        #                                                                    random_state=31 * 9)
        # listoftraintestsplits[i][1], listoftraintestsplits[i][3] = shuffle(listoftraintestsplits[i][1],
        #                                                                    listoftraintestsplits[i][3],
        #                                                                    random_state=31 * 9)

        trainsplitshape = (listoftraintestsplits[i][0].shape[0], 1, listoftraintestsplits[i][0].shape[1])
        testsplitshape = (listoftraintestsplits[i][1].shape[0], 1, listoftraintestsplits[i][1].shape[1])

        listoftraintestsplits[i][0] = listoftraintestsplits[i][0].reshape(trainsplitshape)
        listoftraintestsplits[i][1] = listoftraintestsplits[i][1].reshape(testsplitshape)

    assert len(listoftraintestsplits[0])==6
    '''
    if config_learning.scale:
        sc = preprocessing.StandardScaler().fit(x_train.todense())
        x_train = sc.transform(x_train.todense())
        if x_test is not None: x_test = sc.transform(x_test.todense())
    else:
        sc = None
        x_train = np.array(x_train.todense())
        if x_test is not None: x_test = np.array(x_test.todense())

    y_train_c = to_categorical(y_train, num_classes=num_classes)
    if y_test is not None:
        y_test_c = to_categorical(y_test, num_classes=num_classes)
    else:
        y_test_c = None

    feature_dim = x_train.shape[1]
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    if x_test is not None:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    else:
        x_test = None

    return sc, x_train, x_test, y_train_c, y_test_c, feature_dim