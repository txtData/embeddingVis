# Word Embedding Visualization

**This project visualizes how neural networks learn word embeddings.**

It uses a PyTorch implementation of a Neural Network to learn word embeddings and predict part-of-speech (POS) tags. The network consists of an embedding layer and a linear layer.

The training examples contain sentences where each word is associated with the correct POS tag. The dictionary used for training consists of only ten words: the, a, woman, dog, apple, book, reads, eats, green, and good. The POS tags are noun, verb and adjective.

The goal of the network is to learn a two-dimensional word embedding for each word. In practical applications the dimensionality would of course be higher. Only two dimensions are used to allow easy visualization of the training process on an x/y graph.

Initially, the coordinates of all words are randomly initialized. Through backpropagation, in each training iteration, the neural network adjusts the coordinates of each word slightly. Its aim is to optimize the words'
positions in space so that they become more suitable for predicting the part-of-speech accurately.

&nbsp;

### Visualizing the learning of word embeddings

In the first example, we start with ten words randomly initialized in a 2-dimensional space. Each word is represented by a coordinate in this space. As the model trains, it learns to optimize the word embeddings, adjusting their positions to improve the prediction of part-of-speech (POS) tags.

<img src="videos/gif_no_background.gif" width="600"/>

&nbsp;

### Visualizing the learning of prediction boundaries

In this visualization, you can additionally see how the model learns to classify words into categories according to their POS tags:

<img src="videos/gif_1.gif" width="600"/>

&nbsp;

### Another run, with a very different outcome

The last video example shows that--depending on different random initializations--different representations and classification boundaries are being learned:

<img src="videos/gif_2.gif" width="600"/>


