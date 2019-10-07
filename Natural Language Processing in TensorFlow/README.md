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



## Useful Link

[News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)

[IMDB Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/): You will find here 50,000 movie reviews which are classified as positive of negative.

[Tensorflow Projector](http://projector.tensorflow.org/)

[TFDS Subwords text encoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder)

