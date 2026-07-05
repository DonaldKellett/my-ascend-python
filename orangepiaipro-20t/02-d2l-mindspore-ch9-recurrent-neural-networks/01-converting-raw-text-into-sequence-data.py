import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import urllib.request

"""
01-converting-raw-text-into-sequence-data.py
MindSpore adaptation of D2L chapter 9.2
https://d2l.ai/chapter_recurrent-neural-networks/text-sequence.html
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
    MPLBACKEND = os.getenv('MPLBACKEND')
    print(f'Using Matplotlib backend: {MPLBACKEND}')

    """
    Download a copy of H. G. Wells' "The Time Machine"
    Construct a vocabulary from tokens (words) within the book
    """
    prefix_url = 'https://d2l-data.s3-accelerate.amazonaws.com'
    time_machine_url = f'{prefix_url}/timemachine.txt'
    raw_text = ''
    with urllib.request.urlopen(time_machine_url) as response:
        raw_text = response.read().decode('utf-8')
    text = re.sub('[^A-Za-z]+', ' ', raw_text).lower()
    words = text.split()
    vocab = Vocab(words)

    """
    Plot the unigram (word) frequencies against log-log scale with Matplotlib
    Word frequency follows the Zipfian power law distribution
    Read more about Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """
    freqs = np.array([freq for token, freq in vocab.token_freqs])
    freqs_x = np.arange(freqs.size)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(freqs_x, freqs, label='Word frequencies', linestyle='-')
    plt.title('Word frequencies')
    plt.xlabel('token: x')
    plt.ylabel('frequency: n(x)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(
        '00-ch9-text-sequence-unigram-frequencies.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

    """
    Plot the bigram and trigram frequencies against log-log scale with Matplotlib
    Bigrams are consecutive pairs of words
    Trigrams are consecutive 3-tuples of words
    Like unigrams, their frequencies satisfy Zipf's law albeit with a smaller exponent
    """
    bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    trigram_tokens = ['--'.join(triple) for triple in zip(words[:-2], words[1:-1], words[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    bigram_freqs = np.array([freq for token, freq in bigram_vocab.token_freqs])
    bigram_freqs_x = np.arange(bigram_freqs.size)
    trigram_freqs = np.array([freq for token, freq in trigram_vocab.token_freqs])
    trigram_freqs_x = np.arange(trigram_freqs.size)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(freqs_x, freqs, label='unigram', linestyle='-')
    plt.plot(bigram_freqs_x, bigram_freqs, label='bigram', linestyle='--')
    plt.plot(trigram_freqs_x, trigram_freqs, label='trigram', linestyle='-.')
    plt.title('n-gram frequencies')
    plt.xlabel('token: x')
    plt.ylabel('frequency: n(x)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(
        '01-ch9-text-sequence-n-gram-frequencies.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

if __name__ == '__main__':
    main()
