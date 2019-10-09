### Tokenizer

- ```Tokenizer```:
    - ```num_words```: the maximum number of words to keep, based on word frequency. E.g. ```num_words = 100``` if word not in top 100 frequency, will not included
    - ```oov_token = "<oov>"``` replace out-of-vocabulary words during ```text_to_sequence``` calls 
- ```fit_on_texts```: Updates internal vocabulary based on a list of sequences. 
- ```tokenizer.word_index``` : return dictinary key is words, value is correponding token

```python
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

# sentences of different lengths
sentences = [
     "I love my dog",        
     "I, love my cat",
     "You love my dog!", # ! won't impact token as dog!
     "Do you think my dog is amazing?"
]

tokenizer = Tokenizer(num_words = 100, oov_token = "<oov>")  #oov: out-of-vocabulary
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index # return dictinary key(word) value(token) pair

sequences = tokenizer.texts_to_sequences(sentences) #transform each text in texts into integers from token
print(word_index)
print(sequences)

"""
Output
{'<oov>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
[[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
"""
```

### Padding 

- ```padding```:String, ```'pre'``` or ```'post'```: pad either before or after each sequence.  Default is ```'pre'```
- ```maxlen```: maximum length of all sequences.
  - ```truncating```:   ```'pre'``` or ```'post'```: remove values from sequences larger than ```maxlen```, either at the beginning or at the end of the sequences. Default is ```pre```

E.g. If have sentence less than length of 5, will padding at the end of sentence. If have sentence length longer than 5, will truncate at the end of the sentence 
```python
 pad_sequences(sequences, padding = "post", truncating='post', maxlen = 5)
 ```


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences #for padding 
padded = pad_sequences(sequences)
print(padded)

"""
Output
[[ 0  0  0  5  3  2  4]
 [ 0  0  0  5  3  2  7]
 [ 0  0  0  6  3  2  4]
 [ 8  6  9  2  4 10 11]]
"""

```



#### TFDS Dataset

- install TFDS ```pip install -q tensorflow-datasets```

```python
import tensorflow as tf
print(tf.__version__)

tf.enable_eager_execution() #not necessary if you have TF 2.0 installed

import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
```

**Deal with dataset**

```
import numpy as np

train_data, test_data = imdb['train'], imdb['test'] #25000 fro training, 25000 for testing
training_sentences, training_labels = [], []
testing_sentences, testing_labels = [], []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(str(s.numpy()))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
```

#### Embedding 

-  ```tf.keras.layers.Embedding```: Use specified dimension (input_dim x output_dim) Embedded matrix to get embedded vector for each word in the sequence with length of ```input_length```. Embedded matrix applied to each word is the same. Output will be  input_length x output_dim
    - ```input_dim``` as vocabulary dimension for one-hot vector
    - ```output_dim``` as embedding dimension. 
    - ```input_length```: Length of input sequences, when it is constant. This argument is **required** if you are going to connect ```Flatten``` then ```Dense``` layers upstream(without it, the shape of the dense outputs cannot be computed).

```python
vocab_size = 10000
embedding_dim = 16
max_length = 120

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # parameter is embedding matrix size = vocab_size x embedding_dim, input_length is length of sentence
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation = tf.nn.relu),
    tf.keras.layers.Dense(1, activation = 'sigmoid')                             
])
```

**Train Model**

```python
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, truncating = trunc_type)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(padded, 
          training_labels_final, 
          epochs = 10, 
          validation_data = (testing_padded, testing_labels_final))

```

#### Use TFDS Pre-tained Tokenizer

We use **subwords8k** for **imdb_reviews**, must use **tensorflow version 2.0+**. 

- ```tokenizer = info.features["text"].encoder```: get pre-trained tokenizer
- ```tokenizer.encode(sample_string)```: encode string use pre-trained tokenizer
- ``` tokenizer.decode(tokenized_string)```: encode string use pre-trained tokenizer

```python
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info = True, as_supervised = True)
train_data, test_data = imdb["train"], imdb["test"]

tokenizer = info.features["text"].encoder #access tokenizer from Pre-trained 

print(tokenizer.subwords) #by looking up its vocabulary

sample_string = "TensorFlow, from basics to mastery"

tokenized_string = tokenizer.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print("The original string: {}".format(original_string))
```



## Useful Link

[News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)

[IMDB Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/): You will find here 50,000 movie reviews which are classified as positive of negative.

[Tensorflow Projector](http://projector.tensorflow.org/)

[TFDS Subwords text encoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder)

