# Transfer learning
Transfer learning refers to the process of using a pretrained model as a starting point to tuning your parameters. The premise of transfer learning lies in the idea that models that are trained on lots of data have developed features that are 'transferable' to the new task at hand.

For example, you may want to train a model to classify x-ray images, but you only have access to limited amount of data. You can take a model, trained on a large dataset, and use its weights to initialize your model weights with. All you need to do is to replace the last layer(s) of the model with you own and only train those weights on your data.

Transfer learning usually makes sense when you have limited data for the problem at hand, but have access to a model that is trained for a similar task on a large, but possibly different, dataset. In which case, you can rely on the fact that the two datasets share features that are common between the them that the model may have picked up.