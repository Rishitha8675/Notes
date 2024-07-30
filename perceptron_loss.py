#In this we will use loss function to train the perceptron

from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

x,y = make_classification(n_samples=200, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class = 1, random_state= 41, hypercube= False, class_sep=10)
plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap='winter',s=100)
plt.show()

class perceptron:

   def __init__(self):
     self.learning_rate=0.1
     self.weights=np.ones(x.shape[1]+1)
   
   
   def fit(self,x,y):
   
       x=np.insert(x,0,1,axis=1)
       
       for j in range(1000):
           gradient=0
           for i in range(len(x)):
               
               gradient+=self.gradient_descent(x[i],y[i])
           self.weights+= self.learning_rate*gradient
           
           
       return self.weights    
         
         
     
   def gradient_descent(self,Xi,Yi):
     
     if Yi*(np.dot(Xi,self.weights)) >= 0:
       return 0
     else:
       return -1*Yi*(Xi)  
       
       
classifier=perceptron()
weights = classifier.fit(x,y)
print(weights)       
      
     
  
      
     
      
       
     





