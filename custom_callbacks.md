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













