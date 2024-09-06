
## Huber loss function

```
def my_huber_loss(y_true, y_pred):
   threshold=0.2
   error = y_true-y_pred
   is_small_error=tf.abs(error)<=threshold
   small_error_loss = tf.square(error)/2
   big_error_loss = threshold*(tf.abs(error)-(0.5*threshold))
   return tf.where(is_small_error,small_error_loss,big_error_loss)

```

```
model.compile(loss='my_huber_loss')

```

### Adding hyperparameters to custom loss functions

```
def_my_huber_loss_with_threshold(threshold):
   def my_huber_loss(y_true, y_pred):
     threshold=0.2
     error = y_true-y_pred
     is_small_error=tf.abs(error)<=threshold
     small_error_loss = tf.square(error)/2
     big_error_loss = threshold*(tf.abs(error)-(0.5*threshold))
     return tf.where(is_small_error,small_error_loss,big_error_loss)
   return my_huber_loss  

```

```
model.compile(loss=my_huber_loss_with_threshold) 

``` 
### Turning loss functions into classes

```
from tensorflow.keras.losses import Loss

class MyHuberLoss(Loss):
    threshold=0.5
    def __init__(self,threshold):
       super().__init__()
       self.threshold=threshold
    
    def call(self,y_true_y_pred):
       error=y_true-y_pred
       is_small_error = tf.abs(error) <= self.threshold
       small_error_loss = tf.square(error)/2
       big_error_loss = self.threshold*(tf.abs(error))-(0.5*self.threshold)
       return tf.where(is_small_error,small_error_loss,big_error_loss)

 ```

```
 model.compile(loss=MyHuberLoss(threshold=0.7))

```   
## Contrastive loss function

To calculate the loss in `Siamese Network Architecture` we need a new type of loss function which is not a in-built one.  
  
The idea is if the images are similar, we produce feature vectors that are very simailar and if images are different, produce feature vectors that are dissimilar  

`\( Y \cdot D^2 + (1 - Y) \cdot \max(\text{margin} - D, 0)^2 \)`  
This is the formula for contrastive loss  
`Y` -> is the tensor of details about image similairty  
     `1` if the images are similar and `0` if they are not  
`D` -> is the tensor of eucliadean distances between the pairs of the images  
`margin` -> is the minimum distance that can be present btwn the elements in order to consider them as similar or not  
if `y=1` => Expression will get reduced to `D^2`  
if `y=0` => Expression will get reduced to `max(margin-D,0)^2`  


