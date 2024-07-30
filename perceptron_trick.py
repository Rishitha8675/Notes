from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

x,y = make_classification(n_samples=200, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class = 1, random_state= 41, hypercube= False, class_sep=10)
plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',s=100)
plt.show()




def perceptron(x,y):
   x=np.insert(x,0,1,axis=1)
   weights=np.ones(x.shape[1])
   learning_rate=0.1
   
   for i in range(1000):
     j = np.random.randint(0,100)
     y_hat = step(np.dot(x[j],weights))
     weights = weights + learning_rate*(y[j]-y_hat)*x[j]
     
     
   return weights[0],weights[1:]
   
   
def step(z):
    return 1 if z>0 else 0
    
    
intercept,coeff =  perceptron(x,y)

print(coeff)
print(intercept)     
     
m= coeff[0]/coeff[1]
c = intercept/coeff[1]    
     
     
x_input = np.linspace(-3,3,100)
y_input = m*x_input + c
    

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='red',linewidth=3)
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,2)    
plt.show() 
   
