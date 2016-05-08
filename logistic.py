# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library

def logistic(z):
    return 1/(1+np.exp(-z))

def logistic_derivative(z):
    return logistic(z)*(1-logistic(z))
if __name__ == "__main__":
    # Plot the logistic function
    z = np.linspace(-6,6,100)
    plt.plot(z, logistic(z), 'b-')
#     plt.plot(z, logistic_derivative(z), 'r-')
    plt.xlabel('$z$', fontsize=15)
    plt.ylabel('$\sigma(z)$', fontsize=15)
    plt.title('logistic function')
    plt.grid()
    plt.show()
    
    plt.plot(z, logistic_derivative(z), 'r-')
    plt.xlabel('$z$', fontsize=15)
    plt.ylabel('$\\frac{\\partial \\sigma(z)}{\\partial z}$', fontsize=15)
    plt.title('derivative of the logistic function')
    plt.grid()
    plt.show()
