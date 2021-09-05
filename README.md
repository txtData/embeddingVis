# Word Embedding Visualization

**This project visualizes how neural networks learn word embeddings.**

A Neural Network implemented PyTorch and consisting of an embedding and a linear layer is fed with a small toy set of training examples that contain the correct part-of-speech (POS) tag for each word in the sentence. The used dictionary constists of only ten words: the, a, woman, dog, apple, book, reads, eats, green and good.

The Neural Neural network will learn 2-dimensional word embeddings for each word, with the aim of predicting their POS tags correctly. Since the embedding uses only two dimensions (in a realistic setting, dimensionality would of course be significantly higher), they can easily be visualized in a standard x,y-graph.

The coordinates of all words are initialiazed randomly. Using backpropagation, in each training iteration, the neural network alters the coordinates of each word slightly, trying to optimize their position in space, so that they are more suitable to predict the words' part-of-speech.  

The following animations illustrate the network’s learning process:

In the first example, you can see how ten words are randomly initialized in a 2-dimensional space and how the model then learns to optimize each word's embedding, so that words with the same POS tags are grouped closer together.

<img src="videos/gif_no_background.gif" width="600"/>

In this visualization, you can additionally see how the model learns to classify words into categories according to their POS tags.

<img src="videos/gif_1.gif" width="600"/>

The last video example shows that--depending on different random initializations--different representations and classification boundaries are being learned.

<img src="videos/gif_2.gif" width="600"/>


