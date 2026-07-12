import collections
import matplotlib.pyplot as plt
import mindspore
import mindspore.amp as amp
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.nn as nn
import mindspore.ops as ops
import mlflow
import numpy as np
import os
import re
import urllib.request

from mindspore import dtype as mstype
from mindspore.train import Callback, EarlyStopping, Loss, LossMonitor, Model

"""
00-openbible-kjv-elman-rnn.py
Word-level language model trained on the King James Bible using Elman RNN
Dataset courtesy of Open Bible
https://openbible.com/textfiles/kjv.txt
"""

"""
Define our custom evaluation cell for MindSpore's Model API
This is required since we define a custom training loop
"""
class CustomEvalCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomEvalCell, self).__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        logits = self.backbone(data)
        loss = self.loss_fn(logits, label)
        return loss, logits, label

"""
Custom step-wise training cell with gradient clipping
"""
class CustomTrainStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(CustomTrainStepCell, self).__init__(network, optimizer)
        self.grad_fn = mindspore.value_and_grad(self.network, grad_position=None, weights=self.weights)

    def construct(self, data, label):
        loss, grads = self.grad_fn(data, label)
        clipped_grads = clip_by_global_norm(grads)
        loss = ops.depend(loss, self.optimizer(clipped_grads))
        return loss

"""
MLflow logging callback with perplexity
"""
class MLflowLogging(Callback):
    def __init__(self, run_name, network_type, loss_fn, learning_rate, batch_size, epochs, weight_decay=0.0, momentum=0.0, optimizer='sgd'):
        super().__init__()
        self.run_name = run_name
        self.network_type = network_type
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer = optimizer

        self.run = mlflow.start_run(run_name=self.run_name)
        hyperparameters = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'loss_fn': self.loss_fn,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'network_type': self.network_type,
            'epochs': self.epochs
        }
        mlflow.log_params(hyperparameters)

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        current_loss = cb_params.net_outputs.asnumpy().mean()
        current_perplexity = np.exp(current_loss)
        mlflow.log_metric('train_loss', current_loss, step=cb_params.cur_step_num)
        mlflow.log_metric('train_perplexity', current_perplexity, step=cb_params.cur_step_num)

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if hasattr(cb_params, 'eval_results') and cb_params.eval_results:
            val_loss = cb_params.eval_results.get('loss', 0.0)
            val_perplexity = np.exp(val_loss)
            mlflow.log_metric('val_loss', val_loss, step=cb_params.cur_epoch_num)
            mlflow.log_metric('val_perplexity', val_perplexity, step=cb_params.cur_epoch_num)

    def on_train_end(self, run_context):
        mlflow.end_run()

"""
Word-level language model based on Elman RNN with 2 hidden layers of state
"""
class ElmanRNNLM(nn.Cell):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = 256
        self.hidden_size = 256
        self.num_layers = 2
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = nn.RNN(
            self.embedding_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dense1 = nn.Dense(self.hidden_size, self.vocab_size)

    def construct(self, X):
        batch_size, seq_len = X.shape[0], X.shape[1]
        embs = self.embedding(X)
        h0 = ops.zeros((self.num_layers, batch_size, self.hidden_size), dtype=embs.dtype)
        output, hx_n = self.rnn(embs, h0)
        y = self.dense1(output)
        return y

"""
Custom cross-entropy loss function for our sequence data
"""
class SequenceCrossEntropyLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, logits, labels):
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        labels_flat = labels.view(batch_size * seq_len)

        loss = self.loss(logits_flat, labels_flat)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

"""
Convert tokens into numerical indices for training and inference
Taken straight from D2L chapter 9.2
https://d2l.ai/chapter_recurrent-neural-networks/text-sequence.html
"""
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

"""
Manual implementation of mindspore.ops.clip_by_global_norm to avoid
Ascend 310B1 missing SelectV2 operator issue
"""
def clip_by_global_norm(grads, clip_norm=1.0):
    total_sq = 0.0
    for g in grads:
        arr = g.asnumpy()
        total_sq += np.sum(arr * arr)

    norm = np.sqrt(total_sq)

    if norm > clip_norm:
        scale = clip_norm / norm
        clipped = tuple([mindspore.Tensor(g.asnumpy() * scale, dtype=g.dtype) for g in grads])
        return clipped
    else:
        return grads

"""
Generate a continuation for a given prefix using the trained RNN language model
"""
def predict(network, prefix, num_preds, vocab):
    network.set_train(False)
    network = network._backbone

    words = prefix.lower().split()
    word_indices = [vocab[word] for word in words]

    outputs = [word_indices[0]]

    batch_size = 1
    hidden_size = network.hidden_size
    vocab_size = network.vocab_size
    num_layers = network.num_layers

    emb_dtype = network.embedding(mindspore.Tensor([[0]], dtype=mstype.int32)).dtype
    h = ops.zeros((num_layers, batch_size, hidden_size), dtype=emb_dtype)

    for i in range(len(words) + num_preds - 1):
        cur_idx = outputs[-1]

        idx_tensor = mindspore.Tensor([[cur_idx]], dtype=mstype.int32)
        emb = network.embedding(idx_tensor)
        rnn_output, h = network.rnn(emb, h)

        logits = network.dense1(rnn_output[0])

        if i < len(words) - 1:
            outputs.append(word_indices[i + 1])
        else:
            pred_idx = int(logits.argmax(axis=1).asnumpy()[0])
            outputs.append(pred_idx)

    return ' '.join([vocab.idx_to_token[idx] for idx in outputs])

def transform_ds(dataset, batch_size, num_steps, vocab_size):
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    EPOCHS = os.getenv('EPOCHS', '100')
    # KJV Bible has approx. 750k words, a 1M sample limit includes all words
    MAX_SAMPLES = os.getenv('MAX_SAMPLES', '1000000')
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MPLBACKEND = os.getenv('MPLBACKEND')
    epochs = int(EPOCHS)
    max_samples = int(MAX_SAMPLES)
    print(f'Using MLflow tracking URI: {MLFLOW_TRACKING_URI}')
    print(f'Using Matplotlib backend: {MPLBACKEND}')
    print(f'Training our model over at most {epochs} epochs ...')
    print(f'Limiting the maximum number of samples to {max_samples} ...')

    experiment_name = '03-openbible-kjv-elman-rnn'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    """
    Download a copy of the King James Bible from a local mirror
    """
    prefix_url = 'https://www.donaldsebleung.com/assets/datasets/openbible'
    kjv_url = f'{prefix_url}/kjv.txt'
    raw_text = ''
    with urllib.request.urlopen(kjv_url) as response:
        raw_text = response.read().decode('utf-8-sig')

    """
    Construct a vocabulary and corpus from tokens (words) within the book
    """
    raw_lines = raw_text.split('\r\n')[2:]
    lines = [line.split('\t')[1] for line in raw_lines if line != '']
    text = re.sub('[^A-Za-z]+', ' ', ' '.join(lines)).lower()
    words = text.split()
    vocab = Vocab(words, min_freq=5)
    corpus = [vocab[word] for word in words]

    """
    Plot the n-gram frequencies (n = 1, 2, 3) against log-log scale with Matplotlib
    Verify that the n-gram frequencies follow Zipf's law
    """
    unigram_tokens = words
    unigram_vocab = Vocab(unigram_tokens)
    bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    trigram_tokens = ['--'.join(triple) for triple in zip(words[:-2], words[1:-1], words[2:])]
    trigram_vocab = Vocab(trigram_tokens)

    unigram_freqs = np.array([freq for token, freq in unigram_vocab.token_freqs])
    unigram_freqs_x = np.arange(unigram_freqs.size)
    bigram_freqs = np.array([freq for token, freq in bigram_vocab.token_freqs])
    bigram_freqs_x = np.arange(bigram_freqs.size)
    trigram_freqs = np.array([freq for token, freq in trigram_vocab.token_freqs])
    trigram_freqs_x = np.arange(trigram_freqs.size)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(unigram_freqs_x, unigram_freqs, label='unigram', linestyle='-')
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
        '00-kjv-elman-n-gram-frequencies.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

    """
    Extract sequences of num_steps tokens for our features and labels
    Take the first 80% (approx.) samples as our training set with the remainder as our validation set
    This gives 634k training samples and approx. 158k validation samples
    """
    num_steps = 32
    array = mindspore.Tensor([corpus[i:i+num_steps+1] for i in range(len(corpus) - num_steps)])
    X, y = array[:, :-1], array[:, 1:]
    X, y = X[:max_samples], y[:max_samples]
    num_samples = X.shape[0]
    train_samples = 4 * num_samples // 5
    X_train, y_train = X[:train_samples].asnumpy(), y[:train_samples].asnumpy()
    X_test, y_test = X[train_samples:].asnumpy(), y[train_samples:].asnumpy()

    """
    Initialize our training and validation sets and split them into batches of 2^8=256
    """
    vocab_size = len(vocab)
    batch_size = 256
    train_ds = ds.NumpySlicesDataset(data=(X_train, y_train), column_names=['feature', 'label'], shuffle=True)
    test_ds = ds.NumpySlicesDataset(data=(X_test, y_test), column_names=['feature', 'label'], shuffle=True)
    train_ds = transform_ds(train_ds, batch_size=batch_size, num_steps=num_steps, vocab_size=vocab_size)
    test_ds = transform_ds(test_ds, batch_size=batch_size, num_steps=num_steps, vocab_size=vocab_size)

    """
    Define our neural network for training
    """
    net = ElmanRNNLM(vocab_size)
    net_amp = amp.auto_mixed_precision(network=net, amp_level='O2')

    """
    Use cross-entropy loss and minibatch SGD optimizer
    Note that cross-entropy is exactly log-perplexity
    The logarithm function is monotonic increasing so minimizing cross-entropy is equivalent to minimizing perplexity
    """
    learning_rate = 1.0
    loss_fn = SequenceCrossEntropyLoss(reduction='mean')
    optimizer = nn.SGD(params=net_amp.trainable_params(), learning_rate=learning_rate)

    """
    Define our WithLossCell to wrap our network and loss function
    """
    net_amp_with_loss = nn.WithLossCell(
        backbone=net_amp,
        loss_fn=loss_fn
    )

    """
    Specify our custom step-wise training cell to include gradient clipping
    """
    train_net_amp_with_loss = CustomTrainStepCell(
        network=net_amp_with_loss,
        optimizer=optimizer
    )

    """
    Specify our custom evaluation cell
    """
    net_amp_with_loss_eval = CustomEvalCell(
        backbone=net_amp,
        loss_fn=loss_fn
    )

    """
    Train our model over the specified number of epochs
    """
    run_name = '00-openbible-kjv-elman-rnn'
    network_type = 'elman'
    loss_fn_str = 'perplexity'

    model = Model(
        network=train_net_amp_with_loss,
        metrics={'loss': Loss()},
        eval_network=net_amp_with_loss_eval,
        eval_indexes=[0, 1, 2]
    )
    callbacks = [
        LossMonitor(per_print_times=10),
        MLflowLogging(
            run_name=run_name,
            network_type=network_type,
            loss_fn=loss_fn_str,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs
        ),
        EarlyStopping(
            patience=10,
            verbose=True,
            restore_best_weights=True
        )
    ]
    model.fit(
        epoch=epochs,
        train_dataset=train_ds,
        valid_dataset=test_ds,
        callbacks=callbacks
    )

    """
    Predict the next 100 words based on some predefined input
    """
    continuation = predict(
        network=net_amp,
        prefix='In the beginning',
        num_preds=100,
        vocab=vocab
    )
    print(continuation)

if __name__ == '__main__':
    main()
