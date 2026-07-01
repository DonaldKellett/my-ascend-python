import datetime
import gzip
import mindspore
import mindspore.amp as amp
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops as ops
import mlflow
import os
import urllib.request

from mindspore import dtype as mstype
from mindspore.train import Callback, LossMonitor, Model

"""
00-densely-connected-networks-densenet.py
MindSpore adaptation of D2L chapter 8.7
https://d2l.ai/chapter_convolutional-modern/densenet.html
"""

def conv_block(in_channels, out_channels):
    return nn.SequentialCell([
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3)
    ])

def transform_ds(dataset):
    image_transforms = [
        vision.Resize(size=(96, 96)),
        vision.Rescale(rescale=1/255, shift=0),
        vision.HWC2CHW(),
        transforms.TypeCast(data_type=mstype.float32)
    ]
    label_transforms = [
        transforms.OneHot(num_classes=10),
        transforms.TypeCast(data_type=mstype.float32)
    ]
    dataset = dataset.map(operations=image_transforms, input_columns='image')
    dataset = dataset.map(operations=label_transforms, input_columns='label')
    dataset = dataset.batch(batch_size=128, drop_remainder=False)
    return dataset

def transition_block(in_channels, out_channels):
    return nn.SequentialCell([
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    ])

class DenseBlock(nn.Cell):
    def __init__(self, num_convs, in_channels, grow_channels):
        super(DenseBlock, self).__init__()
        self.net = nn.SequentialCell([
            conv_block(in_channels + i * grow_channels, grow_channels)
            for i in range(num_convs)
        ])

    def construct(self, X):
        for blk in self.net:
            Y = blk(X)
            X = ops.cat((X, Y), axis=1)
        return X

class MLflowFashionMNISTCallback(Callback):
    def __init__(self):
        super().__init__()

        run_name = '00-densely-connected-networks-densenet'
        self.run = mlflow.start_run(run_name=run_name)

        hyperparameters = {
            'learning_rate': 0.01,
            'weight_decay': 0,
            'momentum': 0,
            'loss': 'softmax_cross_entropy',
            'optimizer': 'sgd',
            'batch_size': 128,
            'network_type': 'densenet',
            'epochs': 10
        }
        mlflow.log_params(hyperparameters)

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        current_loss = cb_params.net_outputs.asnumpy().mean()
        mlflow.log_metric('train_loss', current_loss, step=cb_params.cur_step_num)

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if hasattr(cb_params, 'eval_results') and cb_params.eval_results:
            val_loss = cb_params.eval_results.get('loss', 0.0)
            val_accuracy = cb_params.eval_results.get('accuracy', 0.0)
            mlflow.log_metric('val_loss', val_loss, step=cb_params.cur_epoch_num)
            mlflow.log_metric('val_accuracy', val_accuracy, step=cb_params.cur_epoch_num)

    def on_train_end(self, run_context):
        mlflow.end_run()

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    MLFLOW_TRACKING_SERVER_HOST = os.getenv('MLFLOW_TRACKING_SERVER_HOST', 'localhost')
    MLFLOW_TRACKING_SERVER_PORT = os.getenv('MLFLOW_TRACKING_SERVER_PORT', '5000')
    MLFLOW_TRACKING_SERVER_URL = f'http://{MLFLOW_TRACKING_SERVER_HOST}:{MLFLOW_TRACKING_SERVER_PORT}'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER_URL)
    experiment_name = '01-d2l-mindspore-ch8-modern-convolutional-neural-networks'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    dataset_dir = 'data/fashion/'
    os.makedirs(dataset_dir, exist_ok=True)

    prefix_url = 'https://donaldsebleung.com/assets/datasets/fashion-mnist'
    X_train_url = f'{prefix_url}/train-images-idx3-ubyte.gz'
    y_train_url = f'{prefix_url}/train-labels-idx1-ubyte.gz'
    X_test_url = f'{prefix_url}/t10k-images-idx3-ubyte.gz'
    y_test_url = f'{prefix_url}/t10k-labels-idx1-ubyte.gz'

    X_train_path = os.path.join(dataset_dir, 'train-images-idx3-ubyte')
    y_train_path = os.path.join(dataset_dir, 'train-labels-idx1-ubyte')
    X_test_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte')
    y_test_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte')

    with urllib.request.urlopen(X_train_url) as response:
        with open(X_train_path, 'wb') as out_file:
            data_gzip = response.read()
            data = gzip.decompress(data_gzip)
            out_file.write(data)

    with urllib.request.urlopen(y_train_url) as response:
        with open(y_train_path, 'wb') as out_file:
            data_gzip = response.read()
            data = gzip.decompress(data_gzip)
            out_file.write(data)

    with urllib.request.urlopen(X_test_url) as response:
        with open(X_test_path, 'wb') as out_file:
            data_gzip = response.read()
            data = gzip.decompress(data_gzip)
            out_file.write(data)

    with urllib.request.urlopen(y_test_url) as response:
        with open(y_test_path, 'wb') as out_file:
            data_gzip = response.read()
            data = gzip.decompress(data_gzip)
            out_file.write(data)

    train_ds = ds.FashionMnistDataset(dataset_dir=dataset_dir, usage='train', shuffle=True)
    test_ds = ds.FashionMnistDataset(dataset_dir=dataset_dir, usage='test', shuffle=True)

    train_ds = transform_ds(dataset=train_ds)
    test_ds = transform_ds(dataset=test_ds)

    net = nn.SequentialCell([
        # Stem - single convolutional and max-pooling layer as in ResNet
        nn.SequentialCell([
            nn.Conv2d(1, 64, kernel_size=7), # use stride=1 instead of 2 in D2L since our images are 96 by 96 only
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        ]),

        # Body - 4 dense blocks each with 4 convolutions followed by transition layer
        # Growth rate = 32 means each convolution adds 32 channels => each dense block adds 128 channels total
        # Transition layer halves height, width and number of channels simultaneously
        # 1. Input from stem: (n, 64, 48, 48)
        # 2. Module #1 with 4 convolutions followed by transition layer
        #     a. Output from dense block: (n, 64 + 4 * 32 = 192, 48, 48)
        #     b. Output from transition layer: (n, 96, 24, 24)
        # 3. Module #2
        #     a. Output from dense block: (n, 96 + 4 * 32 = 224, 24, 24)
        #     b. Output from transition layer: (n, 112, 12, 12)
        # 4. Module #3
        #     a. Output from dense block: (n, 112 + 4 * 32 = 240, 12, 12)
        #     b. Output from transition layer: (n, 120, 6, 6)
        # 5. Module #4
        #     a. Output from dense block: (n, 120 + 4 * 32 = 248, 6, 6)
        #     b. Output from transition layer: (n, 124, 3, 3)
        nn.SequentialCell([
            DenseBlock(num_convs=4, in_channels=64, grow_channels=32),
            transition_block(in_channels=192, out_channels=96)
        ]),
        nn.SequentialCell([
            DenseBlock(num_convs=4, in_channels=96, grow_channels=32),
            transition_block(in_channels=224, out_channels=112)
        ]),
        nn.SequentialCell([
            DenseBlock(num_convs=4, in_channels=112, grow_channels=32),
            transition_block(in_channels=240, out_channels=120)
        ]),
        nn.SequentialCell([
            DenseBlock(num_convs=4, in_channels=120, grow_channels=32),
            transition_block(in_channels=248, out_channels=124)
        ]),

        # Head - Global average pooling followed by linear layer
        nn.SequentialCell([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dense(124, 10)
        ])
    ])

    net_amp = amp.auto_mixed_precision(network=net, amp_level='O2')
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = nn.SGD(params=net_amp.trainable_params(), learning_rate=0.01)

    model = Model(
        network=net_amp,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={'accuracy', 'loss'}
    )
    callbacks = [
        LossMonitor(per_print_times=10),
        MLflowFashionMNISTCallback()
    ]
    epochs = 10
    model.fit(
        epoch=epochs,
        train_dataset=train_ds,
        valid_dataset=test_ds,
        callbacks=callbacks,
        dataset_sink_mode=False
    )

if __name__ == '__main__':
    main()
