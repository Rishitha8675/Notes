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
  
    
      
        
  
  








