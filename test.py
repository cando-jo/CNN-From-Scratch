import numpy as np
from numpy import nan
import pandas as pd
import csv
################## Reading test set ############################

data_test = pd.read_csv("letters.csv",skiprows=[7,14,21,28,35,42])
data_test = np.array(data_test)
data_test = data_test.reshape(7,6,6)



#################### Reading Data ##############################

########## Max Poooling ############
class Conv3x3:

    def __init__(self, num_filters):
        self.num_filters = num_filters

        self.filters = [[[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]]]

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

def get_pooled_layers(test_x):
    conv = Conv3x3(1)
    pool = MaxPool2()

    pooled_test_inputs = []
    for i in range(len(test_x)):
        output_test = conv.feed(test_x[i])
        output_test = sigmoid_postConv(output_test)
        output_test = pool.feed(output_test)
        pooled_test_inputs.append(output_test.reshape(4))
    pooled_test_inputs = np.array(pooled_test_inputs)
    return pooled_test_inputs


pooled_test = get_pooled_layers(data_test).T

###################################

W1_data = []
W1_datas = []
b1_datas = []
W2_datas = []

data_params = pd.read_csv("parameters.csv",skiprows=None)
params_values = np.array(data_params)
for i in params_values:
    for q in i:
        W1_data.append(q)

W1_data = np.array(W1_data)
W1_data = W1_data[np.logical_not(np.isnan(W1_data))]
for i in range(len(W1_data)):
    if(i < 8):
        W1_datas.append(W1_data[i])
    elif(i > 7 and i < 10):
        b1_datas.append(W1_data[i])
    else:
        W2_datas.append(W1_data[i])

W1_datas, b1_datas, W2_datas = np.array(W1_datas),np.array(b1_datas),np.array(W2_datas)
W1_datas = W1_datas.reshape(2,4)
b1_datas = b1_datas.reshape(2,1)
W2_datas = W2_datas.reshape(2,2)



#########################################################################
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
def forward_prop(W4x2, b1, W2x2, X):
    Z1 = W4x2.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2x2.dot(A1)
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def test_predictions(A2):
    return softmax(A2)

def make_predictions(X, W4x2, b1, W2x2):
    _, _, _, A2 = forward_prop(W4x2, b1, W2x2, X)
    predictions = test_predictions(A2)
    return predictions



dev_predictions = make_predictions(pooled_test, W1_datas, b1_datas, W2_datas)
proc_number,pred_o1,pred_o2 = 0,0,0
classification = ""

t_predictions = dev_predictions.T

for i in t_predictions:
    proc_number += 1
    pred_o1,pred_o2 = i[0],i[1]
    if i[0] > i[1]:
        classification = "O Letter"
    else:
        classification = "X Letter"
    ans = "Processing letter number= " + str(proc_number) + "  Output of NN is " + str(format(pred_o1,".3f")) + "  " + str(format(pred_o2,".3f")) + "  ==>  " + classification
    print(ans)

input('')
