import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

show_background = True
write_movie_file = False
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ImageMagick-7.0.9-Q16\\ffmpeg.exe'

# data used to train the embedding and classifier
training_data = [
    (['the', 'dog', 'eats'], ['DET', 'NN', 'V']),
    (['the', 'dog', 'eats', 'the', 'green', 'apple'], ['DET', 'NN', 'V', 'DET', 'ADJ', 'NN']),
    (['the', 'woman', 'reads', 'a', 'good', 'book'], ['DET', 'NN', 'V', 'DET', 'ADJ', 'NN'])
]

# The words that are displayed in the graph and their colors
display_words = ['the', 'a', 'woman', 'dog', 'apple', 'book', 'reads', 'eats', 'green', 'good']
display_colors    = [0, 0, 1, 1, 1, 1, 2, 2, 3, 3]


# defining PyTorch model architecture
class SimpleTagger(nn.Module):

    def __init__(self, vocab_size, labels, embedding_dim = 2):
        super(SimpleTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, labels)

    def forward(self, sentence):        
        embeddings = self.word_embeddings(sentence)                
        return self.use_embeddings(embeddings), embeddings
    
    def use_embeddings(self, embeddings):     
        tag_space = self.linear(embeddings)
        tag_scores = F.log_softmax(tag_space, dim=1)        
        return tag_scores
    
    
# helper function to convert sequences to the format PyTorch expects   
def prepare_sequence(seq, lookup_table):
    idxs = [lookup_table[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
            
    
# getting the embeddings ready to be plotted
def get_plot_data():
    with torch.no_grad():
        sentence_in = prepare_sequence(display_words, words_to_ids)
        tag_scores, embeddings = model(sentence_in)
        embeddings = embeddings.numpy()
        values, indices = tag_scores.T.max(0)
        same = [i for i, j in zip(display_colors, indices) if i == j]
        x = embeddings[:,0]
        y = embeddings[:,1]
        result = [x, y , display_colors]
        result.append(len(same)/10)
        return result
        
# getting the classification layer ready to be plotted 
def get_plot_background_data():
    with torch.no_grad():
        xy = np.mgrid[-3.5:3.6:0.1, -3.5:3.6:0.1].reshape(2, -1).T
        tag_scores = model.use_embeddings(torch.from_numpy(xy).float())
        values, indices = tag_scores.T.max(0)
        x = xy[:,0]
        y = xy[:,1]
        return [x, y ,indices]
    
# creating the animation
def animate(i):
    plt.cla()
    x1, y1, c1, same, x2, y2, c2 = plots[i]
    if show_background:
        ax.scatter(x2, y2, c=c2, alpha= 0.2, s=80, vmin=0, vmax=3)
    scatter = ax.scatter(x1, y1, c=c1, alpha=1.0, s=100,vmin=0, vmax=3)
    for j in range(len(x1)):
        ax.annotate(' '+display_words[j], (x1[j], y1[j]))
    plt.legend(handles=scatter.legend_elements()[0], labels=tags_to_ids, loc='upper right', fontsize=20)
    plt.xlim(-3.5,3.5)
    plt.ylim(-3.5,3.5)  
    plt.title(f'Epoch: {i}   Accuracy: {same:.0%}', fontsize=25)
    
    
# computing lookup tables
words_to_ids = {}
for sent, tags in training_data:
    for word in sent:
        if word not in words_to_ids:
            words_to_ids[word] = len(words_to_ids)           
tags_to_ids = {}    
for sent, tags in training_data:
    for tag in tags:
        if tag not in tags_to_ids:
            tags_to_ids[tag] = len(tags_to_ids)
ids_to_tags = {v: k for k, v in tags_to_ids.items()}


# setting up the model
model = SimpleTagger(len(words_to_ids), len(tags_to_ids))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# the training loop
plots=[]
for epoch in range(200):
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, words_to_ids)
        targets = prepare_sequence(tags, tags_to_ids)
        tag_scores, embedding = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
    plot = get_plot_data()
    plot.extend(get_plot_background_data())
    plots.append(plot)
     
# showing the animation and writing the animation file        
font = {'size': 20}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(15, 15))    
anim = animation.FuncAnimation(fig, animate, frames=200, interval=100, repeat = True) 
if write_movie_file:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('video.mp4', writer=writer)
    

