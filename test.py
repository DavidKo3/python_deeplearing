from numpy import loadtxt, where, reshape
from pylab import scatter, show, legend, xlabel, ylabel

#load the dataset
data = loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
pos = where(y == 1)

neg = where(y == 0)

scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Admitted', 'Not Admitted'])
show()

a=[[2,3],[4,5]]
b=[1,2,3,4]
print len(a)
print len(b)
c=reshape(b,(len(b),1))

print c