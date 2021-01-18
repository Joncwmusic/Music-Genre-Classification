# Music-Genre-Classification

The goal is to get a computer to predict genre classfication but try to do it better than when I tried this before. Previous results are posted as pdfs in this repository.
I'll use the previous results (floating around 60% accuracy) as a benchmark for now.


1. Collect a whole lot of mp3 or wav files (or at least find a music dataset to trian the model)
  Music Dataset:https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
2. Use librosa to extract features from each audio file and use strictly the first 30 seconds of the file to create the spectrograph image
  Dataset already has this but maybe I'll do it anyways for practice
3. Create some neural network architecture to predict the genre of each song
4. Explore other classification Methods and compare models: LogReg, SVM (with various kernels), LDA, QDA, Random Forest, Neural Network Classification

Updates:
On the CSV dataset of preextracted features:
A Deep neural net with 5 FC layers (57 input -> 20 tanh -> 20 tanh -> 15 tanh -> 12 tanh -> 10 softmax)
got up to 65% accuracy on training and 60% accuracy on testing set. This about par with the graduate research project but was computationally expensive. 
The SVM model also got similar results with tweaked hypterparameters and a raidal kernel
The logistic regression model did not perform as well but is the most efficient
KNN was pretty bad when it came to generalizing to the test set
Other SVM models performed similarly not so good

The following may need to happen:
  1. Acquire more labelled data - Not likely but maybe if I find another dataset with the same genres
  2. Modify NN depth/nodes/activations/hyperparam tuning
  3. Ensemble Methods (Note I actually did a version of this with logistic regression on the 3 second clips where the clips predicted classes become votes for the larger 30 second sample)

Using the Spectrogram Images:
  1. Used a Convolutional Neural Network on the full 30 second spectrogram images.
  2. Images were 218, 336, 4 dimensional and split into train and validation sets for the model
  3. NN architecture: (Conv2D Relu -> Conv2D Relu -> MaxPool2D ->) x3 -> Conv2D Relu -> MaxPool2D -> Dense10 tanh-> Dense12 tanh -> Dense10 softmax
  4. Training accuracy as high as 90% but Test Accuracy at 40% at ~50 epochs. Very prone to overfitting. Horrendously inefficient.

using Raw Audio:
  1. We'll try this next.
