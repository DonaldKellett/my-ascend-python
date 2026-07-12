import collections
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
from mindspore.train import Callback, Loss, LossMonitor, Model

"""
05-concise-implementation-of-recurrent-neural-networks.py
MindSpore adaptation of D2L chapter 9.6
https://d2l.ai/chapter_recurrent-neural-networks/rnn-concise.html
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
Concise implementation of RNN-based character-level language model
"""
class MyRNNLM(nn.Cell):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = 32

        self.rnn = nn.RNN(self.vocab_size, self.hidden_size)
        self.dense1 = nn.Dense(self.hidden_size, self.vocab_size)

    def construct(self, X):
        X = ops.transpose(X, (1, 0, 2))
        seq_len, batch_size = X.shape[0], X.shape[1]
        h0 = ops.zeros((1, batch_size, self.hidden_size))
        output, hx_n = self.rnn(X, h0)
        y = ops.transpose(self.dense1(output), (1, 0, 2))
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
    network = network._backbone

    prefix_indices = [vocab[ch] for ch in prefix]

    outputs = [prefix_indices[0]]

    batch_size = 1
    hidden_size = network.hidden_size
    vocab_size = network.vocab_size

    h = ops.zeros((1, batch_size, hidden_size), dtype=mstype.float32)

    on_value = mindspore.Tensor(1.0, dtype=mstype.float32)
    off_value = mindspore.Tensor(0.0, dtype=mstype.float32)

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
        transforms.TypeCast(data_type=mstype.float32)
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
    EPOCHS = os.getenv('EPOCHS', '100')
    epochs = int(EPOCHS)
    print(f'Using MLflow tracking URI: {MLFLOW_TRACKING_URI}')
    print(f'Training our model over {epochs} epochs ...')

    experiment_name = '02-d2l-mindspore-ch9-recurrent-neural-networks'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

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
    train_ds = ds.NumpySlicesDataset(data=(X_train, y_train), column_names=['feature', 'label'], shuffle=True)
    test_ds = ds.NumpySlicesDataset(data=(X_test, y_test), column_names=['feature', 'label'], shuffle=True)
    train_ds = transform_ds(train_ds, batch_size=batch_size, num_steps=num_steps, vocab_size=vocab_size)
    test_ds = transform_ds(test_ds, batch_size=batch_size, num_steps=num_steps, vocab_size=vocab_size)

    """
    Define our neural network for training
    """
    net = MyRNNLM(vocab_size)
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
    net_amp_with_loss = nn.WithLossCell(backbone=net_amp, loss_fn=loss_fn)

    """
    Specify our custom step-wise training cell to include gradient clipping during backpropagation
    """
    train_net_amp_with_loss = CustomTrainStepCell(
        network=net_amp_with_loss,
        optimizer=optimizer
    )

    """
    Specify our custom evaluation cell
    """
    net_amp_with_loss_eval = CustomEvalCell(backbone=net_amp, loss_fn=loss_fn)

    """
    Train our model over the specified number of epochs using Model API
    """
    run_name = '05-concise-implementation-of-recurrent-neural-networks'
    network_type = 'rnn'
    loss_fn_str = 'softmax_cross_entropy'

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
        )
    ]
    model.fit(
        epoch=epochs,
        train_dataset=train_ds,
        valid_dataset=test_ds,
        callbacks=callbacks,
        dataset_sink_mode=False
    )

    """
    Predict the next 20 tokens based on some predefined input
    """
    continuation = predict(
        network=net_amp,
        prefix='it has',
        num_preds=20,
        vocab=vocab
    )
    print(continuation)

if __name__ == '__main__':
    main()
