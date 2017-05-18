deep neural network model code repository. The DNN model consists of the following modules:

- Trainer 
  The trainer reads prestine images and corresponding training
  input data in the form of class labels and pixel coordinates. 
  The pixel coordinates are used to extract regions of the prestine
  input image as samples to train the model. The class labels
  are used for classiffying the pixels. The trainer outputs training
  weights to a directory where the prediction module can access and
  produce a segmentation of the input image based on the results of 
  the training module.

  NOTE: The model operates on a concatenation of input from all 
        training sessions.  This is necessary to preserve prediction
        results across sessions.

- Predictor
  The prediction module is repsonsible for segmenting an input image
  based on probabilities from the training module.  It reads training
  weights and classifies all the pixels of the input image.  It outputs
  a new image whose pixels are colored according to the class labels
  of the segment they belong to.
