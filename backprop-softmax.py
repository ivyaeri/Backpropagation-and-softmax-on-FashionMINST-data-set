
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s # TODO
def sigmoid_d(x):
    sg=sigmoid(x)
    return (sg*(1-sg)) # TODO
def tanh(x):
    tanh=np.exp(x)-np.exp(-x)
    return tanh/(np.exp(x)+np.exp(-x))
def tanh_d(x):
    t=tanh(x)
    return 1-t**2
def relu(x):
    return np.maximum(0,x)# TODO
def relu_d(x):
    for j in x:
        rel=np.where(j>0,1,0)
    return rel# TODO
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,20,10]):

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = tanh
        self.phi_d         = tanh_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = x/255 - 0.5      #normalize and Center the input values between [-0.5,0.5]
        # TODO
        for i in range(1,len(self.z)):
            self.z[i]= np.dot(self.w[i],self.a[i-1])+self.b[i]
            if i!=self.L-1:
                self.a[i]=self.phi(self.z[i])
            else:
                self.a[i]=self.softmax(self.z[i])
        
        return(self.a[self.L-1])

    def softmax(self, z):
        ep=np.exp(z)
        ep=ep/ep.sum()
        # TODO        
        return ep  

    def loss(self, pred, y):
        return -np.log(pred[np.argmax(y)])
        # TODO
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        # TODO
        self.delta[self.L-1]=self.a[self.L-1]-y
        le=len(self.delta)
        for i in range(le-2,-1,-1):
            cx= self.w[i+1].T@self.delta[i+1]
            self.delta[i]=self.phi_d(self.z[i])*cx
        for i in range(1,self.L):
            self.dw[i]=np.asmatrix(self.delta[i]).T@np.asmatrix(self.a[i-1])
            self.db[i]=self.delta[i]

    # Return predicted image class for input x
    def predict(self, x):
        return np.argmax(self.forward(x))# TODO

    # Return predicted percentage for class j
    def predict_pct(self, j):
        return self.a[self.L-1][j]# TODO 
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=36,
            epsilon=0.16 ,
            epochs=30):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
       
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            print(t)
            print(test_acc_log[t])
            
            if test_acc_log[t]>=0.869:
                break
            
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            print(train_acc_log[t])
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                # TODO
                
                # Mini-batch loop
                
                dc_dw = [i*0 for i in self.dw]
                dc_db = [i*0 for i in self.db]
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    # TODO
                    self.forward(x)
                    # Compute gradients
                    # TODO
                    self.backward(x,y)
                    
                    for l in range(self.L):
                        dc_dw[l] =dc_dw[l]+ self.dw[l]
                        dc_db[l] =dc_db[l]+ self.db[l] 

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                
               
                for l in range(1,self.L):
                    
                    self.w[l] -=epsilon *(dc_dw[l]/batch_size)
                    
                    self.b[l] -=epsilon *(dc_db[l]/batch_size)
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0
                
                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)
        return test_acc_log[t-1] 
           
          

# Start training with default parameters.

def main():
    bp = BackPropagation()
    bp.sgd()
    '''#testing the optimal learning rate
    acc=0
    eps=[0.01,0.02,0.04,0.08,0.16,0.32]
    batch=1
    while batch<=60:
        for i in range(batch):
            t=bp.sgd(epsilon=eps[i],batch_size=batch)
            if acc<t:
                acc=t
                opt_lr=eps
                opt_ba=batch
        print(t)
        print(acc)
        batch*=2'''

if __name__ == "__main__":
    main()
    
