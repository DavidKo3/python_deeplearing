# Python imports
import numpy as np # Matrix and vector computation package
np.seterr(all='ignore') # ignore numpy warning like multiplication of inf
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
from matplotlib import cm # Colormaps


# Define and generate the samples
nb_of_samples_per_class = 20  # The number of sample in each class
red_mean = [-1,0]  # The mean of the red class
blue_mean = [1,0]  # The mean of the blue class
std_dev = 1.2  # standard deviation of both classes
# Generate samples from both classes
x_red = np.random.randn(nb_of_samples_per_class, 2) * std_dev + red_mean
x_blue = np.random.randn(nb_of_samples_per_class, 2) * std_dev + blue_mean

# Merge samples in set of input variables x, and corresponding set of output variables t
X = np.vstack((x_red, x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class,1)), np.ones((nb_of_samples_per_class,1))))








# Define the logistic function
def logistic(z):
    return 1/(1+np.exp(-z))

# Define the neural network function y= 1/(1+numpy.exp(-x*w))
def nn(x,w):
    return logistic(x.dot(w.T))
# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(x,w): 
    return np.around(nn(x,w))

# Define the cost function
def cost_nn(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))


# define the gradient function.
def gradient_nn(w, x, t): 
    return (nn(x, w) - t).T * x

# define the update function delta w which returns the 
#  delta w for each weight in a vector
def delta_w_nn(w_k, x, t, learning_rate):
    return learning_rate * gradient_nn(w_k, x, t)




if __name__ == "__main__":
    
    
    # Plot the cost in function of the weights
    # Define a vector of weights for which we want to plot the cost
    nb_of_ws = 100 # compute the cost nb_of_ws times in each dimension
    ws1 = np.linspace(-5, 5, num=nb_of_ws) # weight 1
#     print ws1
#     ws3 = np.linspace(-5, 5, num=5) # weight 1
#     print ws3
    ws2 = np.linspace(-5, 5, num=nb_of_ws) # weight 2
    ws_x, ws_y = np.meshgrid(ws1, ws2) # generate grid
    cost_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize cost matrix
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_ws):
        for j in range(nb_of_ws):
            cost_ws[i,j] = cost_nn(nn(X, np.asmatrix([ws_x[i,j], ws_y[i,j]])) , t)
    # Plot the cost function surface
    plt.contourf(ws_x, ws_y, cost_ws, 20, cmap = cm.pink)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('$\\xi$', fontsize=15)
    plt.xlabel('$w_1$', fontsize=15)
    plt.ylabel('$w_2$', fontsize=15)
    plt.title('Cost function surface')
    plt.grid()
    plt.show()
    # Set the initial weight parameter
    w = np.asmatrix([-4, -2])
    print w
    # Set the learning rate
    learning_rate = 0.05
    # Start the gradient descent updates and plot the iterations
    nb_of_iterations = 10  # Number of gradient descent updates
    w_iter=[w]
    
    for i in range(nb_of_iterations):
        dw = delta_w_nn(w, X, t, learning_rate)  # Get the delta w update
        w = w-dw  # Update the weights
        w_iter.append(w)  # Store the weights for plotting
   
    # Plot the first weight updates on the error surface
    # Plot the error surface
    plt.contourf(ws_x, ws_y, cost_ws, 20, alpha=0.9, cmap=cm.pink)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('cost')
    
    # Plot the updates
    for i in range(1, 4): 
        w1 = w_iter[i-1]
        w2 = w_iter[i]
        # Plot the weight-cost value and the line that represents the update
        plt.plot(w1[0,0], w1[0,1], 'bo')  # Plot the weight cost value
        plt.plot([w1[0,0], w2[0,0]], [w1[0,1], w2[0,1]], 'b-')
        plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'.format(i), color='b')
    w1 = w_iter[3]  
    # Plot the last weight
    plt.plot(w1[0,0], w1[0,1], 'bo')
    plt.text(w1[0,0]-0.3, w1[0,1]+0.7, '$w({})$'.format(4), color='b') 
    # Show figure
    plt.xlabel('$w_1$', fontsize=15)
    plt.ylabel('$w_2$', fontsize=15)
    plt.title('Gradient descent updates on cost surface')
    plt.grid()
    plt.show()
   