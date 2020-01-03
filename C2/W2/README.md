# Data augmentation
Data augmentation is often used in computer vision problems to enlarge the dataset. The premise of data augmentation is that the new augmented data may introduce features to the training set that can help unocover features in the validation/test set that do not currently exist in - and hence cannot be learned from - the training set. 

Data augmentation is effectively a method to reduce overfitting. 

Data augmentation methods include:
- Flip (horizontal, vertical)
- Shear
- Zoom
- Rotate
- Shift

## When is data augmentation helpful
Data augmentation introduces certain randomness to the training set. This randomness may help improve the validation/test performance if the validation/test set has the same randomness. However, if the validation/test set does not have the same randomness introduced by data augmentation to the training set, then data augmentation does not translate to improved validatino/test performance. 

As an example, say you are training a shoe classifer. All your images in both the training and validation sets only contain shoes pointing to the right. Now if you augment your training set by horizontally flipping your images, and effectively create pictures of shoes pointing to the left, you will likely not gain validation performance because there is no shoe pointing to the left in your validation set. 

The message here is that augmentation is only helpful if it can introduce elements of the validation set that are currently absent from the training set. The `horse_human_augmentation.py` is an example of a case when augmentation does not help improve validation performance. 
