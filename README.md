# Adversarial-AES

※This code is under development.

This is an AES model using a domain adversarial neural network.

## Requirements
- python = 3.8.10
- tensorflow = 2.9.2
- numpy = 1.21.6
- pandas = 1.3.5
- nltk = 3.7

## Training
If you want to use word series as an input, you should download `glove.6B.50d.txt` from https://nlp.stanford.edu/projects/glove/ and place in the `embeddings` folder.

Also, download `mnistm.h5` from https://github.com/sghoshjr/tf-dann/releases/download/v1.0.0/mnistm.h5 if you want to try domain adaptation on the MNIST dataset and place in the `Datasets/MNIST_M` folder.

* MNIST/MNIST_M domain adaptation
  * Run `train_sampleDANN.py`.

* ASAP domain adaptation
  * Run the `train_Smodel.py` script. you can chose options: --train_mode _domain-adaptation/source_, --with_features _yes/no_, --input_seq _words/pos_.
