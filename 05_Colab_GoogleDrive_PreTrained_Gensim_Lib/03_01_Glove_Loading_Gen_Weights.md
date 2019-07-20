
- 참고: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

``` python
#4-1. Pre-Trained Word Embedding Vector Loading
embeddings_index = {}

f = open(os.path.join('/content/gdrive/My Drive/AI', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 100 # glove dimension
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
print(embedding_matrix[10, :])

#make model
model = keras.Sequential()
embedding_layer = keras.layers.Embedding(len(tokenizer.word_index) + 1,   #in  dim
                            EMBEDDING_DIM,                                #out dim
                            embeddings_initializer='glorot_uniform',      #glorot_uniform: Xavier uniform initializer.
                            weights=[embedding_matrix],
                            input_length=200,
                            trainable=False) # trainable=True > Fine-Tune, True일때 overfitting
```
