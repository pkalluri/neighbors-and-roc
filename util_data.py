import numpy as np

"""Get shuffled version of X, Y"""
def shuffle(X, Y):
    m = len(X)
    order_of_ids = np.arange(m)
    np.random.shuffle(order_of_ids)
    X = X[order_of_ids]
    Y = Y[order_of_ids]
    return X, Y

"""Returns given X,Y split into two based on the given percentage"""
def split_in_2(SPLIT_PERCENTAGE, X, Y):
    m = len(X)
    num_split1_samples = int(SPLIT_PERCENTAGE * m)
    x_1 = X[:num_split1_samples]
    y_1 = Y[:num_split1_samples]
    x_2 = X[num_split1_samples:]
    y_2 = Y[num_split1_samples:]
    return (x_1, y_1), (x_2, y_2)

"""Returns given X,Y split into 3 based on the given percentages"""
def split_in_3(TRAIN_PERCENTAGE, DEV_PERCENTAGE, X, Y):
    m = len(X)
    num_train_samples = int(TRAIN_PERCENTAGE * m)
    num_dev_samples = int(DEV_PERCENTAGE * m)
    x_train = X[:num_train_samples]
    y_train = Y[:num_train_samples]
    x_dev = X[num_train_samples:num_train_samples + num_dev_samples]
    y_dev = Y[num_train_samples:num_train_samples + num_dev_samples]
    x_test = X[num_train_samples + num_dev_samples:]
    y_test = Y[num_train_samples + num_dev_samples:]
    return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

"""Returns given X,Y split into 3 based on the given percentages"""
def split_list_in_3(TRAIN_PERCENTAGE, DEV_PERCENTAGE, l):
    m = len(l)
    num_train_samples = int(TRAIN_PERCENTAGE * m)
    num_dev_samples = int(DEV_PERCENTAGE * m)
    train_text = l[:num_train_samples]
    dev_text = l[num_train_samples:num_train_samples + num_dev_samples]
    test_text = l[num_train_samples + num_dev_samples:]
    return train_text, dev_text, test_text