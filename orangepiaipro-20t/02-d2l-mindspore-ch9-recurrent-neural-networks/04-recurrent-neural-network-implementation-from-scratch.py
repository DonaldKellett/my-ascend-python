import collections
import mindspore
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

"""
04-recurrent-neural-network-implementation-from-scratch.py
MindSpore adaptation of D2L chapter 9.5
https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html
"""

"""
Recurrent neural network layer from scratch
Here, we implement a simplified version of mindspore.nn.RNN
https://www.mindspore.cn/docs/en/r2.9.0/api_python/nn/mindspore.nn.RNN.html
"""
class MyRNN(nn.Cell):
    def __init__(self, input_size, hidden_size, dtype=mstype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.sigma = 0.01

        self.W_xh = mindspore.Parameter(ops.randn(self.input_size, self.hidden_size, dtype=self.dtype) * self.sigma)
        self.W_hh = mindspore.Parameter(ops.randn(self.hidden_size, self.hidden_size, dtype=self.dtype) * self.sigma)
        self.b_h = mindspore.Parameter(ops.zeros(self.hidden_size, dtype=self.dtype))
        self.tanh = nn.Tanh()

    """
    Here we assume num_layers=1 and num_directions=1
    So we simplify the hidden state hx to have shape (batch_size, hidden_size) instead
    The returned hx_n also has shape (batch_size, hidden_size)
    """
    def construct(self, x, hx):
        seq_len, batch_size = x.shape[0], x.shape[1]
        output = ops.zeros((seq_len, batch_size, self.hidden_size), dtype=self.dtype)
        hx_n = hx
        for i in range(seq_len):
            hx_n = self.tanh(ops.matmul(x[i], self.W_xh) + ops.matmul(hx_n, self.W_hh) + self.b_h)
            output[i] = hx_n
        return output, hx_n

"""
RNN-based character-level language model from scratch
"""
class MyRNNLM(nn.Cell):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = 32

        self.rnn = MyRNN(self.vocab_size, self.hidden_size, dtype=mstype.float16)
        self.dense1 = nn.Dense(self.hidden_size, self.vocab_size, dtype=mstype.float16)

    def construct(self, X):
        X = ops.transpose(X, (1, 0, 2))
        seq_len, batch_size = X.shape[0], X.shape[1]
        h0 = ops.zeros((batch_size, self.hidden_size), dtype=mstype.float16)
        output, hx_n = self.rnn(X, h0)
        y = ops.zeros((batch_size, seq_len, self.vocab_size), dtype=mstype.float16)
        for i in range(seq_len):
            y[:, i, :] = self.dense1(output[i])
        return y

"""
Custom cross-entropy loss function for our sequence data
"""
class SequenceCrossEntropyLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.loss = nn.SoftmaxCrossEntropyWithLogits(reduction='none')

    def construct(self, logits, labels):
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        labels_flat = labels.view(batch_size * seq_len, vocab_size)
        
        loss = self.loss(logits_flat, labels_flat)
        
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

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

    prefix_indices = [vocab[ch] for ch in prefix]

    outputs = [prefix_indices[0]]

    batch_size = 1
    hidden_size = network.hidden_size
    vocab_size = network.vocab_size

    h = ops.zeros((batch_size, hidden_size), dtype=mstype.float16)

    on_value = mindspore.Tensor(1.0, dtype=mstype.float16)
    off_value = mindspore.Tensor(0.0, dtype=mstype.float16)

    for i in range(len(prefix) + num_preds - 1):
        cur_idx = outputs[-1]

        one_hot = ops.one_hot(
            mindspore.Tensor([cur_idx], dtype=mstype.int32),
            vocab_size,
            on_value=on_value,
            off_value=off_value
        )
        X = one_hot.reshape(1, 1, vocab_size)

        rnn_output, h = network.rnn(X, h)

        logits = network.dense1(rnn_output[0])

        if i < len(prefix) - 1:
            outputs.append(prefix_indices[i + 1])
        else:
            pred_idx = int(logits.argmax(axis=1).asnumpy()[0])
            outputs.append(pred_idx)

    return ''.join([vocab.idx_to_token[idx] for idx in outputs])

def transform_ds(dataset, batch_size, num_steps, vocab_size):
    feature_transforms = [
        transforms.OneHot(num_classes=vocab_size),
        transforms.TypeCast(data_type=mstype.float16)
    ]
    # For sequence data the labels are just features shifted by 1 time step
    label_transforms = feature_transforms
    dataset = dataset.map(operations=feature_transforms, input_columns='feature')
    dataset = dataset.map(operations=label_transforms, input_columns='label')
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    print(f'Using MLflow tracking URI: {MLFLOW_TRACKING_URI}')

    experiment_name = '02-d2l-mindspore-ch9-recurrent-neural-networks'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    """
    Check the output shape of my self-defined RNN layer
    """
    batch_size, input_size, hidden_size, seq_len = 2, 16, 32, 100
    print(f'batch_size={batch_size}')
    print(f'input_size={input_size}')
    print(f'hidden_size={hidden_size}')
    print(f'seq_len={seq_len}')

    my_rnn = MyRNN(input_size, hidden_size, dtype=mstype.float16)
    x = ops.ones((seq_len, batch_size, input_size), dtype=mstype.float16)
    h0 = ops.zeros((batch_size, hidden_size), dtype=mstype.float16)
    print(f'x shape: {x.shape}')
    print(f'h0 shape: {h0.shape}')

    output, hn = my_rnn(x, h0)
    print(f'output shape: {output.shape}')
    print(f'hn shape: {hn.shape}')


    """
    Check the output shape of my RNN-based character-level language model
    """
    batch_size, vocab_size, seq_len = 2, 16, 100
    print(f'batch_size={batch_size}')
    print(f'vocab_size={vocab_size}')
    print(f'seq_len={seq_len}')

    my_rnnlm = MyRNNLM(vocab_size)
    X = ops.ones((batch_size, seq_len, vocab_size), dtype=mstype.float16)
    print(f'X shape: {X.shape}')

    y = my_rnnlm(X)
    print(f'y shape: {y.shape}')

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
    vocab_size = len(vocab)
    corpus = [vocab[token] for token in tokens]

    """
    Extract sequences of num_steps tokens for our features and labels
    Take the first 75% (approx.) samples as our training set with the remainder as our validation set
    This gives 130k training samples and approx. 43k validation samples
    """
    num_steps = 32
    array = mindspore.Tensor([corpus[i:i+num_steps+1] for i in range(len(corpus) - num_steps)])
    X, y = array[:, :-1], array[:, 1:]
    X_train, y_train = X[:130000].asnumpy(), y[:130000].asnumpy()
    X_test, y_test = X[130000:].asnumpy(), y[130000:].asnumpy()

    """
    Initialize our training and validation sets and split them into batches of 2^10=1024
    """
    batch_size = 1024
    batches_per_epoch = 127 # Precomputed based on size of training set
    train_ds = ds.NumpySlicesDataset(data=(X_train, y_train), column_names=['feature', 'label'], shuffle=True)
    test_ds = ds.NumpySlicesDataset(data=(X_test, y_test), column_names=['feature', 'label'], shuffle=True)
    train_ds = transform_ds(train_ds, batch_size=batch_size, num_steps=num_steps, vocab_size=vocab_size)
    test_ds = transform_ds(test_ds, batch_size=batch_size, num_steps=num_steps, vocab_size=vocab_size)

    """
    Define our neural network for training
    """
    net = MyRNNLM(vocab_size)

    """
    Use cross-entropy loss and minibatch SGD optimizer
    Note that cross-entropy is exactly log-perplexity
    The logarithm function is monotonic increasing so minimizing cross-entropy is equivalent to minimizing perplexity
    """
    learning_rate = 1.0
    loss_fn = SequenceCrossEntropyLoss(reduction='mean')
    optimizer = nn.SGD(params=net.trainable_params(), learning_rate=learning_rate)

    """
    Define our forward and gradient functions
    """
    def forward(X, y):
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        return loss, y_hat

    grad_fn = mindspore.value_and_grad(fn=forward, grad_position=None, weights=optimizer.parameters, has_aux=True)

    """
    Define the training logic on a single batch and epoch
    """
    def train_batch(X_batch, y_batch):
        (loss, _), grads = grad_fn(X_batch, y_batch)
        # Clip gradients to avoid the exploding gradients issue common in RNNs
        clipped_grads = clip_by_global_norm(grads)
        optimizer(clipped_grads)
        return loss

    def train_epoch(epoch=0):
        print(f'Epoch {epoch} start')
        batch_count = train_ds.get_dataset_size()
        net.set_train()
        for batch_idx, (X_batch, y_batch) in enumerate(train_ds.create_tuple_iterator()):
            loss = train_batch(X_batch, y_batch)
            loss_ndarray = loss.asnumpy()
            perplexity_ndarray = np.exp(loss_ndarray)
            if batch_idx % 10 == 0:
                print(f'Training loss: {loss_ndarray:.4f} [{batch_idx}/{batch_count}]')
            mlflow.log_metric('train_loss', loss_ndarray, step=epoch*batches_per_epoch+batch_idx)
            mlflow.log_metric('train_perplexity', perplexity_ndarray, step=epoch*batches_per_epoch+batch_idx)
        print(f'Epoch {epoch} end')

    """
    Define the validation logic at the end of each epoch
    """
    def validate_epoch(epoch=0):
        validation_losses = []
        net.set_train(False)
        for X_batch, y_batch in test_ds.create_tuple_iterator():
            batch_size = X_batch.shape[0]
            y_hat = net(X_batch)
            validation_loss = (batch_size, loss_fn(y_hat, y_batch).item())
            validation_losses.append(validation_loss)
        val_samples_total = sum(batch_size for batch_size, _ in validation_losses)
        val_loss = sum(batch_size * batch_loss for batch_size, batch_loss in validation_losses) / val_samples_total
        val_perplexity = np.exp(val_loss)
        print(f'Validation loss after epoch {epoch}: {val_loss:.4f}')
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        mlflow.log_metric('val_perplexity', val_perplexity, step=epoch)

    """
    Train our character-level language model over 100 epochs
    """
    run_name = '04-recurrent-neural-network-implementation-from-scratch'
    network_type = 'rnn'
    loss_fn_str = 'softmax_cross_entropy'
    epochs = 100
    weight_decay = 0.0
    momentum = 0.0
    optimizer_str = 'sgd'

    run = mlflow.start_run(run_name=run_name)
    hyperparameters = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'loss_fn': loss_fn_str,
        'optimizer': optimizer_str,
        'batch_size': batch_size,
        'network_type': network_type,
        'epochs': epochs
    }
    mlflow.log_params(hyperparameters)

    print(f'Training our model over {epochs} epochs ...')
    for epoch in range(epochs):
        train_epoch(epoch=epoch)
        validate_epoch(epoch=epoch)

    mlflow.end_run()

    """
    Predict the next 20 tokens based on some predefined input
    """
    continuation = predict(
        network=net,
        prefix='it has',
        num_preds=20,
        vocab=vocab
    )
    print(continuation)

if __name__ == '__main__':
    main()
