import numpy as np
from  MLP_class import network
from numpy.random import seed, rand, uniform, randint 
from matplotlib import pyplot as plt


# function for training the network
def train(net1, ip, op, batch_size, n_iter,  eta,  l2_norm_lambda):
##    for mini_batch in mini_batches:
##        Db, Dw = backprop(mini_batch)
##        b = b - eta*Db
##        w = w - eta*Dw

    #create mini batches
    mini_batches_index = []
    mini_batches_x = []
    mini_batches_y = []
    for i in range(n_iter):
        mini_batches_index.append(randint(0,len(ip),batch_size))
        x = []
        y = []
        for j in range(batch_size):
            x.append(ip[mini_batches_index[i][j]])
            y.append(op[mini_batches_index[i][j]])
        mini_batches_x.append(x)
        mini_batches_y.append(y)

    #mini batch training
    iter1 = []
    loss1 = []
    for i in range(n_iter):
        net1.mean_loss, net1.weights,net1.bias,  net1.mean_weight_derivatives, net1.mean_bias_derivatives  = net1.batch_sgd_backprop(mini_batches_x[i], mini_batches_y[i],l2_norm_lambda)
        print (" mini batch : " , i , " loss : " , net1.mean_loss)
        loss1.append(net1.mean_loss)
        iter1.append(i+1)
        for i in range(len(net1.weights)):
            #update weights and bias
            net1.weights[i] = net1.weights[i] - eta* net1.mean_weight_derivatives[i]
            net1.bias[i] = net1.bias[i] - eta* net1.mean_bias_derivatives[i].squeeze()
    print("new net1.weights : " , net1.weights)
    print("new net1.bias : ", net1.bias)
    return iter1, loss1

#function to test the network
def test(net1, test_ip, test_op):
    print("test ip : ", test_ip)
    print("test op : ", test_op)
    predict = []
    for i in range(len(test_ip)):
        predict.append(net1.feed_forward(test_ip[i]))
    print(len(test_op), len(predict))
    for i in range(len(test_op)):
        print("i : ", i , " ip : ", test_ip[i], " predicted : ", predict[i], " test_op : ", test_op[i])
    return predict

#mean squared error
def mse(actual, predicted):
    return np.mean(np.square(actual - predicted))

#root mean squared error
def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual - predicted)))

#main function to design network, training and testing it
def main():
    seed(1)
    n = 100 # no of exapmles to be generated
    #first data point
    #x1  = uniform(0.0, 10.0, n)
    x1 = [i for i in range(1,101)]
    x1 = np.array(x1)
    ##x1  = randint(0, 100, n)
    seed(17)
    print("x1 : ", x1)
    #second data point
    #x2 = uniform(0.0,10.0, n)
    ##x2 = randint(0,10, n)
    x2 = [i for i in range(11,111)]
    x2 = np.array(x2)   
    ip = np.zeros((n,2))
    for i in range(n):
        ip[i][0] = x1[i]
        ip[i][1] = x2[i]
    print(x2)
    # output variable
    y = x1*x2
    y = y #scaling output for NN
    op = np.array(y)
    print(y)
    #no. of neurons in MLP network 
    # input1[0] - input size , fixed to 2
    # input1[-1] - output size , fixed to 1
    # input[1:-1] - (feel free to change and test)
    # neurons in middle layers, you can use any numbers layers and any number of input neurons in each layer
    
    
    input1  = [2, 8, 8 , 1]

    #create network 
    net1 = network(input1)

    #print initial network summary
    net1.print_network()

    #hyperparameters, tuned for this network feel free to change
    l2_norm_lambda= 0.01
    batch_size = 32
    n_iter = 100
    eta = 0.0000005

    #start training
    iter1, loss1 = train(net1, ip, op, batch_size, n_iter, eta, l2_norm_lambda)

    #generating test dataset
    seed(100)
    m = 100
    test_x1  = uniform(0, 10, m)
    ##test_x1  = randint(0, 10, m)
    seed(64)
    test_x2 = uniform(0,10, m)
    ##test_x2 = randint(0,10,m)

    test_ip = np.zeros((m,2))
    for i in range(m):
        test_ip[i][0] = test_x1[i]
        test_ip[i][1] = test_x2[i]

    # output variable
    test_y = test_x1 * test_x2

    #scaling output variable
    test_y = test_y
    test_op = np.array(test_y)

    #testing the network 
    predicted = test(net1, ip, op)

    #get accuracy parameters for tested network
    print("total mean square error for test set : ", mse(op, predicted))
    print("total root mean square error for test set : ", rmse(op, predicted))

    #plot loss vs iteration graph 
    plt.plot(iter1,loss1)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("train loss graph")
    #plt.show()
    plt.savefig("train loss graph")
# calling main function
main()


    

