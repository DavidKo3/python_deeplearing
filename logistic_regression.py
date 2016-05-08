# Python imports
import numpy as np # Matrix and vector computation package
np.seterr(all='ignore') # ignore numpy warning like multiplication of inf
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
from matplotlib import cm # Colormaps

# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)

# Define and generate the SAMPLES

nb_of_samples_per_class = 20
red_mean = [-1, 0] #the mean of the red class
blue_mean = [1, 0] # the mean of the blue class
std_dev = 1.2
# Generate samples from both classes
x_red = np.random.randn(nb_of_samples_per_class,2)*std_dev + red_mean
x_blue = np.random.randn(nb_of_samples_per_class,2)*std_dev + blue_mean

# Merge samples in set of input variables x, and corresponding set of output variables t
x= np.vstack((x_red, x_blue))
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
def cost(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))





if __name__ == "__main__":
    # Plot both classes on the x1, x2 plane
    plt.plot(x_red[:,0], x_red[:,1], 'ro', label='class red')
    plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='class blue')
    plt.grid()
    plt.legend(loc=2)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.axis([-4, 4, -4, 4])
    plt.title('red vs. blue classes in the input space')


    # Plot the cost in function of the weights
    # Define a vector of weights for which we want to plot the cost
    nb_of_ws = 100 # compute the cost nb_of_ws times in each dimension
    ws1 = np.linspace(-5, 5, num=nb_of_ws) # weight 1
    print ws1
    
    
    plt.show()

