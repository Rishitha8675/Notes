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
z=tf.tensordot(x,y,axes=1)  
z=tf.reduce_sum(x*y,axis=0)  
z=x**5  





