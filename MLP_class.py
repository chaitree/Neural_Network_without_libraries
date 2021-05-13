import numpy as np
# used matrix muliplications in forward and backward propogation

class network(object):

    def __init__(self, lst=[]):

        #create network using inputs given 
        self.ip_size = lst[0]
        self.op_size = lst[-1]
        self.no_of_layers = len(lst[1:-1])
        self.neurons = lst[1:-1]
        self.weights = []
        self.bias = []
        # randomly initialize weights and bias
        for i in range(1,len(lst)):
            wt = np.random.uniform(0.0, 0.05,(lst[i],lst[i-1]))
            self.weights.append(wt)
            self.bias.append(np.random.uniform(0.0, 0.5, lst[i]))

    #print network 
    def print_network(self):
        print("input size : ", self.ip_size)
        print("output_size : ", self.op_size)
        print("no of layers : ", self.no_of_layers)
        for i in range(len(self.weights)):
            if (i == self.no_of_layers):
                print("output layer initial weights : ", self.weights[i])
                print("output layer initial bias : ", self.bias[i])
            else:
                print("layer " , i+1, " initial weights : ", self.weights[i])
                print("layer " , i+1, " initial bias : ", self.bias[i])
        return

    #activations ReLu used in the the middle layers
    def ReLu_activation(self, ip):
        op = np.zeros(ip.shape[0])
        for i in range(ip.shape[0]):
           op[i] = max(0, ip[i])
        return op

    # mean squared error as a cost function. (actual - predicted)**2
    def cost_function(self, exp, y):
        self.mse = (np.square(exp - y)).mean(axis=0)
        return self.mse

    #derivative of cost function used in the network, 2*(actual - predicted)
    def derivative_cost_function(self, exp, y, l2_norm_lambda):
        loss1 = 2*(exp - y)
        l2_norm = (l2_norm_lambda/2)*(np.mean(self.weights[-1])) #L2 regularization
        return loss1 + l2_norm

    # ReLu derivative for backpropogation, 1 if ip > 0 , 0 if ip<=0
    def derivative_relu_function(self,ip):
        op = np.zeros(ip.shape)
        for i in range(ip.shape[0]):
            if ( ip[i] > 0):
                    op[i] = 1
            else:
                    op[i]=0
        return op


    # feed forward equations for MLP
    def feed_forward(self,ip):
        "forward propagation "
        self.layer_inputs = []
        self.layer_inputs.append(ip)
        self.activations = []
        self.activations.append(ip)
        for i in range(len(self.weights)):
            #w*x + b
            self.layer_inputs.append(np.matmul(self.weights[i], self.layer_inputs[i])+self.bias[i])
            # activation(w*x+b)
            self.activations.append(self.ReLu_activation(self.layer_inputs[i+1]))
        feed_forward_op = np.array(self.activations[-1])
        return feed_forward_op
        

    def batch_sgd_backprop(self,X,Y,l2_norm_lambda = 0.001):
        "X-input array"
        "Y-output array"
        "Update weights"
        loss_all = []
        weight_derive_all = []
        bias_derive_all = []
        for i in range(len(X)):
            # collect outputs for every ip
            loss, weights, bias, weight_derive, bias_derive = self.single_example_backprop(X[i], Y[i], l2_norm_lambda)
            loss_all.append(loss)
            weight_derive_all.append(weight_derive)
            bias_derive_all.append(bias_derive)
        self.mean_loss = np.mean(np.array(loss_all), axis=0)
        self.mean_weight_derivatives = []
        self.mean_bias_derivatives = []

        # get mean of outputs from batches
        for j in range(len(weight_derive_all[0])):
            ls = []
            ls1 = []
            for i in range(len(weight_derive_all)):
                ls.append(weight_derive_all[i][j])
                ls1.append(bias_derive_all[i][j])
            self.mean_weight_derivatives.append(np.mean(ls, axis = 0))
            self.mean_bias_derivatives.append(np.mean(ls1, axis = 0)) 

        return self.mean_loss, self.weights, self.bias, self.mean_weight_derivatives, self.mean_bias_derivatives 

    # back propogation for one ip example
    def single_example_backprop(self,x,y, l2_norm_lambda = 0.001):
        self.loss = []
        self.delta = []
        feed_forward_op = self.feed_forward(x)
        self.loss  = self.cost_function(y,feed_forward_op)
        for i in range(self.no_of_layers):
            self.delta.append(np.zeros(self.neurons[i]))
        self.delta.append(self.derivative_cost_function(y, feed_forward_op, l2_norm_lambda))
        for i in range(self.no_of_layers-1, -1, -1):

            del1 = np.matmul(np.transpose(self.weights[i+1]),self.delta[i+1])
            del2 = del1  * self.derivative_relu_function(self.layer_inputs[i+1])
            # local gradient for all the layers
            self.delta[i] = del2

        self.weight_derivatives = []
        self.bias_derivatives = []
        for i in range(len(self.weights)):
            self.weight_derivatives.append(np.zeros(self.weights[i].shape))
            self.bias_derivatives.append(np.zeros(self.weights[i].shape[0]))
        for i in range(len(self.weights)-1, -1, -1):
            # derivative w.r.t weights and bias
            del1 = np.matmul(self.delta[i].reshape(self.delta[i].shape[0],1), np.transpose(self.activations[i].reshape(self.activations[i].shape[0],1)))
            self.weight_derivatives[i] = del1
            self.bias_derivatives[i] = self.delta[i].reshape(self.delta[i].shape[0],1)

        return self.loss, self.weights, self.bias, self.weight_derivatives, self.bias_derivatives
