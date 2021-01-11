# Music-Genre-Classification

The goal is to get a computer to predict genre classfication but try to do it better than when I tried this before. Previous results are posted as pdfs in this repository.
I'll use the previous results (floating around 60% accuracy) as a benchmark for now.


1. Collect a whole lot of mp3 or wav files (or at least find a music dataset to trian the model)
  Music Dataset:https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
2. Use librosa to extract features from each audio file and use strictly the first 30 seconds of the file to create the spectrograph image
  Dataset already has this but maybe I'll do it anyways for practice
3. Create some neural network architecture to predict the genre of each song
4. Explore other classification Methods and compare models: LogReg, SVM (with various kernels), LDA, QDA, Random Forest, Neural Network Classification
5. A few more Ideas: Ensemble Classifier i.e. getting classfiers to 'vote' on their predictions. 
   Also,Since the GTZAN Dataset is already parsed into 3 second segments of larger 30 second samples, it may be better to classifiy the 3 second samples and predict those genres and use the largest vote of the 3 second predictions to then predict the 30 second genre.
   So in other words a blues track parsed into 10 3 second segments might be classified as "blues, rock, blues, blues, metal, hip hop, blues, pop, blues, blues" would be classified as blues.
