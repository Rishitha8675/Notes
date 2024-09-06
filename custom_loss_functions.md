
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

