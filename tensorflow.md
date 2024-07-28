# Tensors  


## Initialization of Tensors  

### 0-D Tensor  
 `x=tf.constant(4)` 
  creates a 0-D tensor `4`

### 1-D Tensor  
` x=tf.constant(4,shape=(1,1),dtype=tf.float32)`
 creates a 1-D tensor `[4]`  
we can mention shape attribute or it is set by default  
we can mention dtype attribute or it is set as float32 by default 
### 2-D Tensor
`
 x=tf.constant([[1,2,3],[4,5,6]])
`

`x=tf.ones((3,3))` creates a tensor of shape=(3,3) 