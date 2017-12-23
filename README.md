# Best_algorithm_for_my_thesis
Some trials to determinate the best machine learning algorithm for my thesis

I try with 3 different machine learning techniques:
-Neural_network (MLP)
-kNN_regression
-SVM_regression

I'm using as feature vector 100 discrete points get from each waveform [x.npy] (only a part of the inspiral phase and the ringdown phase) as it is visible in the following image:

![alt text](https://github.com/ASantosMorales/Best_algorithm_for_my_thesis/blob/master/feature_vectors.png)

Every discrete waveform has its corresponding mass ratio [mass_ratio-npy].

After a systematic evaluation, I found that the best technique for my thesis is a Neural_network with one hidden layer with 25 neurons with sigmoidal activation function.
