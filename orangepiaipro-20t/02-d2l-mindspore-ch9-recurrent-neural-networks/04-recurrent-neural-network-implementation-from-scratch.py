import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindspore import dtype as mstype

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
        seq_len, batch_size = X.shape[0], X.shape[1]
        h0 = ops.zeros((batch_size, self.hidden_size), dtype=mstype.float16)
        output, hx_n = self.rnn(X, h0)
        y = ops.zeros((batch_size, seq_len, self.vocab_size), dtype=mstype.float16)
        for i in range(seq_len):
            y[:, i, :] = self.dense1(output[i])
        return y

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

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
    X = ops.ones((seq_len, batch_size, vocab_size), dtype=mstype.float16)
    print(f'X shape: {X.shape}')

    y = my_rnnlm(X)
    print(f'y shape: {y.shape}')

if __name__ == '__main__':
    main()
