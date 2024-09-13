## Lambda Layers


```python
from keras import backend as K

def my_relu(x):
   return K.maximum(0,x)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Lambda(my_relu),
    #above two layers in combine is similar to tf.keras.layers.Dense(128,activation='relu')
    tf.keras.layers.Dense(10,activation='softmax'),
])   

```
## Architecture of a Custom Layer
  
It is a class that collects parameters that encapsulates state and computation to achieve tha layers purpose in a neural network  
`State` implies `weights`  
`computation` typically implies `forward pass` i.e taking an input and do some computation and generate outputs `Y=WX+b`  

```python
class SimpleDense(Layer):
    def __init__(self,units=32):
        super(SimpleDense,self).__init__()
        self.units=units

    def build(self,input_shape) #Create the state of the layer(weights)
         W_init = tf.random_normal_initializer()
         self.W = tf.Variable(name="kernal",initial_value=W_init(shape=(input_shape[-1],self.units),dtype='float32'),
         trainable=True)

         b_init=tf.zeros_initializer()
         self.b=tf.Variable(name="bias",initial_value=b_init(shape=(self.units,),dtype='float32')trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs,self.w) +self.b    

```

### Training a neural network with the custom layer

```python

 model = tf.keras.Sequential([SimpleDense(units=1)])
 model.compile(optimizer='sgd',loss='mean_squared_error')
 model.fit(x,y,epoch=500,verbose=1)
 print(model,predict(10))
 
 ```

 ### Adding activation attribute to a layer

 ```python
class SimpleDense(Layer):
     def __init__(self,units=32,activation=None):
         super(SimpleDense , self).__init__()
         self.units= units
         self.activation = tf.keras.activations.get(activation)

     def call(self,inputs):
        return self.activation(tf.matmul(inputs,self.w)+self.b)    

```
### Implementation
```python 
model = tf.keras.sequential([
    ...  ,
    SimpleDense(128,activation='relu'),
    ....
])
```








