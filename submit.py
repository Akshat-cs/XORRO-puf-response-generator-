# import numpy as np
# import sklearn


import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length


################################
# Non Editable Region Starting #
################################
def my_fit(Z_train):
    size = Z_train.shape[0] - 1
    model = np.empty((16, 16), dtype=object)
    for i in range(16):
        for j in range(16):
            model[i][j] = LogisticRegression(C=1000, max_iter=10000, solver="newton-cg")
    R_train = Z_train[0:size, 0:64]
    S1_train = Z_train[0:size, 64:68]
    S2_train = Z_train[0:size, 68:72]
    y_train = Z_train[0:size, 72:73]

    S1_train_decimal = np.empty((size,))
    for i in range(size):
        sum = 0
        for j in range(4):
            sum = sum + (S1_train[i][j]) * 2 ** (3 - j)
        S1_train_decimal[i] = int(sum)

    S2_train_decimal = np.empty((size,))
    for i in range(size):
        sum = 0
        for j in range(4):
            sum = sum + S2_train[i][j] * 2 ** (3 - j)
        S2_train_decimal[i] = int(sum)

    d = {}
    for i in range(16):
        for j in range(16):
            d[(i, j)] = np.empty((1, 65), dtype=int)
    for i in range(size):
        d[(S1_train_decimal[i], S2_train_decimal[i])] = np.vstack(
            (
                d[(S1_train_decimal[i], S2_train_decimal[i])],
                [np.append(R_train[i], y_train[i])],
            )
        )

    for i in range(16):
        for j in range(16):
            d[(i, j)] = d[(i, j)][1:, :]

    for i in d:
        if d[i].shape[0] == 0:
            continue
        model[i].fit(d[i][:, :-1], d[i][:, -1])
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to train your model using training CRPs
    # The first 64 columns contain the config bits
    # The next 4 columns contain the select bits for the first mux
    # The next 4 columns contain the select bits for the second mux
    # The first 64 + 4 + 4 = 72 columns constitute the challenge
    # The last column contains the response

    return model  # Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict(X_tst, model):
    size = X_tst.shape[0]
    R_test = X_tst[:, 0:64]
    S1_test = X_tst[:, 64:68]
    S2_test = X_tst[:, 68:72]
    y_test = X_tst[:, 71:72]
    S1_test_decimal = np.empty((size,))
    for i in range(size):
        sum = 0
        for j in range(4):
            sum = sum + S1_test[i][j] * 2 ** (3 - j)
        S1_test_decimal[i] = int(sum)

    S2_test_decimal = np.empty((size,))
    for i in range(size):
        sum = 0
        for j in range(4):
            sum = sum + S2_test[i][j] * (2 ** (3 - j))
        S2_test_decimal[i] = int(sum)
    t = {}
    for i in range(16):
        for j in range(16):
            t[(i, j)] = np.empty((1, 65), dtype=int)
    for i in range(size):
        t[(S1_test_decimal[i], S2_test_decimal[i])] = np.vstack(
            (
                t[(S1_test_decimal[i], S2_test_decimal[i])],
                [np.append(R_test[i], y_test[i])],
            )
        )

    for i in range(16):
        for j in range(16):
            t[(i, j)] = t[(i, j)][1:, :]

    pred = np.zeros(R_test.shape[0])
    for i in range(R_test.shape[0]):
        pred[i] = model[(int(S1_test_decimal[i]), int(S2_test_decimal[i]))].predict(
            [R_test[i]]
        )
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to make predictions on test challenges
    # print(pred/total)
    return pred
