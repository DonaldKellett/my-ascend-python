import collections
import mindspore
import mindspore.dataset as ds
import re
import urllib.request

"""
02-language-models.py
MindSpore adaptation of D2L chapter 9.3
https://d2l.ai/chapter_recurrent-neural-networks/language-model.html
"""

# Taken straight from D2L chapter 9.2
# Convert tokens into numerical indices for training and inference
class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    """
    Download a copy of H. G. Wells' "The Time Machine"
    """
    prefix_url = 'https://d2l-data.s3-accelerate.amazonaws.com'
    time_machine_url = f'{prefix_url}/timemachine.txt'
    raw_text = ''
    with urllib.request.urlopen(time_machine_url) as response:
        raw_text = response.read().decode('utf-8')

    """
    Tokenize the text to build the corpus and vocabulary
    """
    text = re.sub('[^A-Za-z]+', ' ', raw_text).lower()
    tokens = list(text)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]

    """
    Extract sequences of num_steps tokens for our features and labels
    Take the first 75% (approx.) samples as our training set
    This gives 130k training samples
    """
    num_steps = 10
    array = mindspore.Tensor([corpus[i:i+num_steps+1] for i in range(len(corpus) - num_steps)])
    X, y = array[:, :-1], array[:, 1:]
    X_train, y_train = X[:130000].asnumpy(), y[:130000].asnumpy()

    """
    Initialize our dataset from the training set and split them into batches of 2
    Inspect a random batch taken from our training set
    """
    batch_size = 2
    train_ds = ds.NumpySlicesDataset(data=(X_train, y_train), column_names=['feature', 'label'], shuffle=True)
    train_ds = train_ds.batch(batch_size=batch_size)
    X_train_batch, y_train_batch = next(train_ds.create_tuple_iterator())
    print(f'X_train_batch: {X_train_batch}')
    print(f'y_train_batch: {y_train_batch}')

if __name__ == '__main__':
    main()
