import numpy as np
from numpy import nan
import pandas as pd
import csv

data_train = pd.read_csv("train.csv",skiprows=[7])

data_train = np.array(data_train).reshape(2,6,6)





letters = data_train


test_y = np.array([[0],[1],[1],[0],[0],[1]])

x = np.array(letters)
y = np.array([[0],[1]])



class Conv3x3:

    def __init__(self, num_filters):
        self.num_filters = num_filters

        self.filters = [[[0,  0, 1],
  [0, 1,  0],
  [ 1, 0,  0]]]

    def iterate_regions(self, image):

        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def feed(self, input):

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

class MaxPool2:

  def iterate_regions(self, image):

    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def feed(self, input):

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output

def sigmoid_postConv(x):

    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def get_pooled_layers():
    conv = Conv3x3(1)
    pool = MaxPool2()
    pooled_inputs = []
    for i in range(len(x)):
        output = conv.feed(x[i])
        output = sigmoid_postConv(output)
        output = pool.feed(output)
        pooled_inputs.append(output.reshape(4))
    pooled_inputs = np.array(pooled_inputs)


    return pooled_inputs


X_train = get_pooled_layers()



X_train = np.array(X_train).T
Y_train = np.array([0,1])


def init_params():
    W4x2 = np.random.rand(2, 4) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2x2 = np.random.rand(2, 2) - 0.5
    return W4x2, b1, W2x2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def forward_prop(W4x2, b1, W2x2, X):
    Z1 = W4x2.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2x2.dot(A1)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W4x2, W2x2, X, Y):

    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    derivative_W2x2 = 1 / m * dZ2.dot(A1.T)
    dZ1 = W2x2.T.dot(dZ2) * deriv_sigmoid(Z1)
    derivative_W4x2 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return derivative_W4x2, db1, derivative_W2x2

def update_params(W4x2, b1, W2x2,derivative_W4x2, db1, derivative_W2x2, alpha):
    W4x2 = W4x2 - alpha * derivative_W4x2
    b1 = b1 - alpha * db1
    W2x2 = W2x2 - alpha * derivative_W2x2
    return W4x2, b1, W2x2

def get_predictions(A2):
    # print("Prediction is:",end='')
    return np.argmax(A2, 0)

def test_predictions(A2):

    return softmax(A2)

def get_accuracy(predictions, Y):
    print(predictions, Y)

    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W4x2, b1, W2x2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W4x2, b1, W2x2,  X)
        derivative_W4x2, db1, derivative_W2x2 = backward_prop(Z1, A1, Z2, A2, W4x2, W2x2, X, Y)
        W4x2, b1, W2x2 = update_params(W4x2, b1, W2x2, derivative_W4x2, db1, derivative_W2x2, alpha)
        if i % 1000 == 0:
            print("Iteration:", i)
            predictions = get_predictions(A2)
            # print("Accuracy is:",get_accuracy(predictions, Y))
    return W4x2, b1, W2x2

w1_beforeTraining, b1_beforeTraining, w2_beforeTraining = init_params()


W4x2, b1, W2x2 = gradient_descent(X_train, Y_train, 0.1, 20000)



def make_predictions(X, W4x2, b1, W2x2):
    _, _, _, A2 = forward_prop(W4x2, b1, W2x2, X)
    predictions = test_predictions(A2)
    return predictions

def test_prediction(index, W4x2, b1, W2x2):
    prediction = make_predictions(X_train[:, index, None], W4x2, b1, W2x2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

#---------------- Storing Data --------------

with open('parameters.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([8,8,8,8])
    # write the header
    writer.writerows(W4x2)
    writer.writerows(b1)
    writer.writerows(W2x2)
###############################################
print("Training done, loaded parameters to parameters.csv")
input('')
