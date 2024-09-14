### Some built in call-backs

* ModelCheckpoint
```python
  model.fit(train_batches,epochs=5,validation_data=validation_batches,verbose=2,callbacks=[ModelCheckpoint('model.h5',save_weights_only=True,verbose=1)]) 
  # If we only want to save the optimal value, we can specify save_best_only=True instead of save_weights_only=True
```
* EarlyStopping 

```python
   model.fit(train_batches,epochs=5,validation_data=validation_batches,verbose=2,callbacks=[EarlyStopping(patience=3,restore_best_weights=True,monitor='val_loss')])

#restore_best_weights uses the best weights as the ultimate ones   

```   

## Custom callbacks

### Callback that displays the time at the start and end of the training

```python
import datatime

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self,batch,logs=None):
        print('Training:batch{}begins at {}'.format(batch,datetime.datatime.now.time()))

    def on_train_batch_end(self,batch,logs=None):
        print('Training: batch{}ends at {}'.format(batch,datetime.datetime.now().time()))    
```

```python
my_custom_callback = MyCustomCallback()

model.fit(x_train,y_train,batch_size=64,epochs=1,verbose=0,callbacks=[my_custom_callback])
```

### Callback that detect overfitting

```python
class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self,threshold):
        super(DetectOverfittingCallback,self).__init__()
        self.threshold = threshold

    def on_epoch_end(self,epoch,logs=None):
        ratio = logs["val_loss"]/logs["loss"]    
        print("Epoch:{}, val/Train loss ratio: {:.2f}".format(epoch,ratio))

    if ratio>threshold:
        print("Stopping training....")    
        self.model.stop_training=True

model.fit(...,callbacks=[DetectOverfittingCallback(threshold=1.3)])

```
### Building VisCallback

```python
#At the end of every epoch, it displays the predicted output and save it
class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self,inputs,ground_truth,display_freq=10,n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images=[]
        self.display_freq=display_freq
        self.n_samples=n_samples

    def on_epoch_end(self,epoch,logs=None):
        #Randomly sample data
        indexes = np.random.choice(len(self.inputs),size=self.n_samples)
        x_test , y_test = self.inputs[indexes],self.ground_truth[indexes]
        predictions=np.argmax(self.model.predict(x_test),axis=1)
        
        display_digits(x_test,predictions,y_test,epoch,n=self.display_freq)

        buf =io.BytesIO()
        plt.savefig(buf,format='png')
        buf.seek(0)
        image=Image.open(buf)
        self.images.append(np.array(image))

        if epoch % self.display_freq == 0:
            plt.show()

        def on_train_end(self,logs=None):
            imageio.mimsave('animation.gif',self.images,fp=1)
```
```python
model.fit(...,callbacks=[VisCallback(x_test,y_test)])
```              














