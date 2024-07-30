
# To avoid tensorflow messages
```
import os  
import tensorflow  
  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  
```



# Tensors  


## Initialization of Tensors  

### 0-D Tensor or Scalar  
 `x=tf.constant(4)` 
  creates a 0-D tensor `4`

### 1-D Tensor or Vector  
` x=tf.constant([1,2,3],shape=(3,),dtype=tf.float32)`
 creates a 1-D tensor `[1,2,3]`  
we can mention shape attribute  
Here it is 1-D tensor and 3-D vector  
In general we don't need to explicitly mention the shape  
we can mention dtype attribute or it is set as float32 by default 

### 2-D Tensor or Matrices
`x=tf.constant([[1,2,3],[4,5,6],[7,8,9],[11,12,13]])`
creates a 2-D tensor of shape (4,3)  

### Other ways
`x=tf.ones((3,3))` creates a 2-D tensor of shape=(3,3)  
`x=tf.zeros((2,3))` creates a 2-D tensor of shape=(2,3)  
`x=tf.eye(3)` creates an identity matrix of 3*3  
`x=tf.random.normal((3,3),mean=0,stddev=1)` creates a 2-D tensor of shape=(3,3) where the values are drawn from a normal distribution with a mean of 0 and a standard deviation of 1  
`x=tf.random.uniform((1,3),minval-0,maxval=1)` creates a 2-D tensor of shape (1,3)  where the values are drawn from a uniform distribution between 0 (inclusive) and 1 (exclusive)  
`x=tf.range(9)` creates a 1-D tensor of values ranging from 0(inclusive) and 9(exclusive) i.e `[0 1 2 3 4 5 6 7 8]`  
`x=tf.range(start=1, limit=10, delta=2)` creates a 1-D tensor i.e `[1 3 5 7 9]`

`x=tf.cast(x,dtype=tf.float64)` converts the datatype of the elements in the tensor x into float  


## Mathematical operations  
```
x= tf.constant([1,2,3])  
y=tf.constant([9,8,7])  
z=tf.add(x,y)
z=x-y
print(z)  
```
will output `tf.Tensor([10 10 10],shape=(3,),dtype=int32)`  
Element wise operation is being done  
```
z= tf.subtract(x,y)  
z=x-y  
  
z=tf.divide(x,y)  
z=x/y  
  
z=tf.multiply(x,y)  
z=x*y  
  
z=tf.tensordot(x,y,axes=1) [For dot product]  
z=tf.reduce_sum(x*y,axis=0)  
  
z=x**5  

```

For matrix Multiplication:
```
x=tf.random.normal((2,3))  
y=tf.random.normal((3,4))  
z=tf.matmul(x,y)  
z=x@y  
```

## Indexing

`x=tf.constant([0,1,2,4,2,1,3,1,1])`  
`print(x[:])` prints all the elements  
`print(x[1:])` prints the elements from the index "1"(inclusive)  
`print(x[1:3])` prints the elements from index "1" and index "3" (exclusive)  
`print(x[::2])` prints `[0 2 2 3 1]` i.e adds +2 to the each element  
'print(x[::-1])` prints the elements in reverse order i.e `[1 1 3 1 2 4 2 1 0]`  

```
indices = tf.constant([0,3])  
x_ind=tf.gather(x,indices)  
print(x_ind)  
```
prints the elements in x with indices we mentioned  

## Reshaping

```
x=tf.range(9)  
x=tf.reshape(x,(3,3))  
```
Reshapes the x which is an 1-D tensor into 2-D tensor of shape (3,3)  

```
x=tf.transpose(x,perm=[1,0])  
```
Here, perm=[1, 0] means to swap the first dimension (rows) with the second dimension (columns)  
  
  
  
  
# Neural Network  
### Sequential API  
```
import tensorflow as tf  
from tensorflow import keras  
model = keras.Sequential(  
    [   keras.Input(shape=(28*28))
        layers.Dense(512,activation='relu'),  
        layers.Dense(256,activation='relu'),  
        layers.Dense(10),  

    ]
)  
print(model.summary())  
model.compile(  
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  
    optimizer=keras.optimizers.Adam(lr=0.001),  
    metrics=["accuracy"],  

)  
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)  
model.evaluate(x_test, y_test, batch_size=32,verbose=2)  
```  
A neural network model trained and tested over the data using Sequential API.  
We can also do  
```
model = keras.Sequential()  
model.add(keras.Input(shape=(784)))  
model.add(layers.Dense(512,activation='relu'))  
model.add(layers.Dense(256,activation='relu'))  
model.add(layers.Dense(10))  
```  
Sequential API is very convenient, not very flexible  

### Functional API

Functional API is a bit more flexible than that of Sequential 
Functional API used to construct a non-linear neural network  


```
from keras.models import Model  
inputs=keras.Input(shape=(734,))  
x=layers.Dense(512, activation='relu')(inputs)  
x=layers.Dense(256,activation='relu')(x)  
outputs=layers.Dense(10,activation='softmax')(x)  
model=Model(inputs=inputs,outputs=outputs)  
    
print(model.summary())  

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  
    optimizer=keras.optimizers.Adam(lr=0.001),  
    metrics=['accuracy'],  
)  
model.fit(x_train,y_train,batch_size=31,epochs=5,verbose=2)  
model,evaluate(x_test,y_test,batch_size=32,verbose=2)  

```  
Structure of a NN using Functional APIS : 
```
from keras.models import Model  
from keras import *  
x=Input(shape=(3,))  
hidden1=Dense(128,activation='relu')(x)  
hidden2=Dense(64,activation='relu')(hidden1)  
output1=Dense(1,activation='linear')(hidden2)  
output2=Dense(2,activation='sigmoid')(hiddden2)  
model=Model(inputs=x,outputs=[output1,output2])  
model.summary()  
  
from keras.utils import plot_model  #displays the architecure of our model
plot_model(model,show_shapes=True)  
```

```
from keras.layers import *  
from keras.models import Model  
  
#define two sets of inputs  
inputA = Input(shape=(32,))  
inputB=Input(shape=(128,))  

#the first branch operates on the first input  
x=Dense(8,activation='relu')(inputA)  
x1=Dense(4,activation='relu')(x)  

#the second branch operates on the second input  
y=Dense(64,activation='relu')(inputB)  
y1=Dense(32,activation='relu')(y)  
y2=Dense(4,activation=;relu)(y1)  

#combine the output of the two branches  
combined=concatenate([x1,y2])  

#apply a FC layer and then a regression prediction on the combined outputs  
z=Dense(2,activation='rely')(combined)  
z1=Dense(1,activation='relu)(z)  

#our model will accept the inputs of the two branches and then output a single value  
model= Model(inputs=[inputA, inputB],outputs=z)  

from keras.utils import plot_model  
plot_model(model) 

``` 

## CNN with Sequential and Functional API  
  
### Using Sequential  
``` 
model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),  
        layers.Conv2D(32,3,padding='valid',activation='relu'),  
        layers.MaxPooling2D(pool_size=(2,2)),  
        layers.Conv2D(64,3,activation='relu'),  
        layers.MaxPooling2D(),  
        layers.Conv2D(128,3,activation='relu'),  
        layers.Flatten(),  
        layers.Dense(64,activation='relu'),  
        layers.Dense(10),
    ]
)
``` 

### Using Functional

```
def my_model():  
   inputs=keras.Input(shape=(32,32,3))  
   x=layers.Conv2D(32,3)(inputs)
   x=layers.BatchNormalization()(x)  
   x=keras.activations.relu(x)  
   x=layers.MaxPoolong2D()(x)  
   x=layers.Conv2D(64,5,padding='same')(x)  
   x=layers.BatchNormalization()(x)  
   x=keras.activations.relu(x)  
   x=layers.BatchNormalization()(x)  
   x=keras.activations.relu(x)  
   x=layers.Flatten()(x)  
   x=layers.Dense(64,activation='relu')(x)  
   outputs=layers.Dense(10)(x)  
   model=keras.Model(inputs=input,outputs=output)  
   return model  
```

### Regularisation and adding Drop out layers
```
from  tensorflow.keras import layers,regularizers  

x=layers.Conv2D(32,3,padding='same',kernal_regularizer=regularizers.l2(0.01),)(inputs)  
x=layers.Dense(64,activation='relu',kernal_regularizers.l2(0.01),)(x)  
x=layers.Dropout(0.5)  
```
In general, we train the model using by epochs if we add drop out layers  



## RNN  

```
model=keras.Sequential()  
model.add(keras.Input(shape=(None,28)))  
model.add(
    layers.SimpleRNN(512,return_sequences=True,activation='relu')
)  
model.add(layers.SimpleRNN(512,activation='relu'))  
model.add(layers.Dense(10))  

```  

## GRU  
```
model=keras.Sequential()  
model.add(keras.Input(shape=(None,28)))  
model.add(
    layers.GRU(256,return_sequences=True,activation='tanh')
)  
model.add(layers.GRU(256,activation='tanh'))  
model.add(layers.Dense(10))  

```  

## LSTM 

```
model=keras.Sequential()  
model.add(keras.Input(shape=(None,28)))  
model.add(
    layers.Bidirectional(layers.LSTM(256,return_sequences=True,activation='tanh'))
)  
model.add(layers.Birectional(layers.LSTM(256,activation='tanh')))  
 model.add(layers.Dense(10))  

```  






  






