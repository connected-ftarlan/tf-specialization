from tensorflow.keras.callbacks import Callback


class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        limit = 0.998
        if logs.get('acc') is not None and logs.get('acc') > limit:
            self.__operation(epoch, limit)
        elif logs.get('accuracy') is not None and logs.get('accuracy') > limit:
            self.__operation(epoch, limit)

    def __operation(self, epoch, threshold):
        if epoch + 1 > 1:
            print('\nTrained for {} epochs'.format(epoch + 1))
        else:
            print('\nTrained for {} epoch'.format(epoch + 1))
        print('Achieved accuracy of >{}%'.format(threshold * 100))
        print('Stopping training...')
        self.model.stop_training = True