
import numpy as np
import matplotlib.pyplot as plt
# Define the vector of input samples as X, with
# 20 values sampled from a uniform distribution between 0 and 1

x = np.random.uniform(0,1 , 20)

#Generae the target values t from x with small gaussian noise so the estimation won`t
# be perfect.
# Define a function f that represnets the line that generates t without noise

def f(x):
    return x*2

# Create the target t with some gaussian noise
noise_variance = 0.2 # Variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = np.random.randn(x.shape[0])*noise_variance

# Create targets t
t= f(x) + noise
print noise

print x.shape[0] 
print x
print t


# Define the neural network function y = x * w
def nn(x, w):
    return x*w 

# Define the cost function
def cost(y, t):
    return ((y-t)**2).sum()

# define the gradient function. Remember that y = nn(x,w) = x*w
def graident(w,x ,t):
    return 2*x*( nn(x,w) - t )

# define the update function delta w
def delta_w(w_k, x, t , learning_rate ):
    return learning_rate * graident(w_k, x, t).sum()


# Set the initial weight parameter
w = 0.1
# Set the learning rate
learning_rate = 0.1

# Start performing the gradienimport matplotlib as pltt descent updates, and print and weights

nb_of_iterations = 4
w_cost = [(w, cost(nn(x,w),t))] # LIst to store the weights, cost values

for i in range(nb_of_iterations):
    dw = delta_w(w,x,t, learning_rate)
    w = w - dw
    w_cost.append((w, cost(nn(x,w),t)))
    print cost(x,w)


for i in range(0, len(w_cost)):
    print ("w({}): {:.4f}\t cost:{:.4f}".format(i, w_cost[i][0], w_cost[i][1]))
    
    
    # gradient dessecents
    
    w = 0
    # Start performing the gradient descent updates.
    nb_of_iterations = 10
    for i in range(nb_of_iterations):
        dw = delta_w(w,x,t,learning_rate)
        print "dw : " , dw
        w = w -dw
        
# Plot the fitted line agains the target line
# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
print f(0), f(1)
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel("input x")
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs target')
plt.grid()
plt.legend(loc=2)
plt.show()




