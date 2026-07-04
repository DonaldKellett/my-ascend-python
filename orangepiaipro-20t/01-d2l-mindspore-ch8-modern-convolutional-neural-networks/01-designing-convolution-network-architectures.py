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
01-designing-convolution-network-architectures.py
MindSpore adaptation of D2L chapter 8.8
https://d2l.ai/chapter_convolutional-modern/cnn-design.html
"""

def transform_ds(dataset, batch_size):
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
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset

class MLflowFashionMNISTCallback(Callback):
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

class ResNeXtBlock(nn.Cell):
    """
    ResNeXt block used by RegNetX
    """
    def __init__(self, in_channels, out_channels, stride=1, group=1):
        super().__init__()
        self.trunk = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, group=group),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ])
        residual = []
        if in_channels != out_channels or stride != 1:
            residual.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
            residual.append(nn.BatchNorm2d(out_channels))
        self.residual = nn.SequentialCell(residual)
        self.relu = nn.ReLU()

    def construct(self, X):
        return self.relu(self.trunk(X) + self.residual(X))

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    print(f'Using MLflow tracking URI: {MLFLOW_TRACKING_URI}')

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

    learning_rate = 0.05
    batch_size = 128
    epochs = 10

    train_ds = ds.FashionMnistDataset(dataset_dir=dataset_dir, usage='train', shuffle=True)
    test_ds = ds.FashionMnistDataset(dataset_dir=dataset_dir, usage='test', shuffle=True)

    train_ds = transform_ds(dataset=train_ds, batch_size=batch_size)
    test_ds = transform_ds(dataset=test_ds, batch_size=batch_size)

    net = nn.SequentialCell([
        # Stem - 3x3 convolution with 32 output channels
        # Input shape: (n, 1, 96, 96)
        # Output shape: (n, 32, 48, 48)
        nn.SequentialCell([
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ]),

        # Body stage 1
        # 1. ResNeXt blocks (depth): 4
        # 2. Output channels:  32
        # 3. Group size: 16 => split channels into 2 groups
        # Input shape: (n, 32, 48, 48)
        # Output shape: (n, 32, 24, 24)
        nn.SequentialCell([
            ResNeXtBlock(32, 32, stride=2, group=2),
            ResNeXtBlock(32, 32, group=2),
            ResNeXtBlock(32, 32, group=2),
            ResNeXtBlock(32, 32, group=2)
        ]),

        # Body stage 2
        # 1. ResNeXt blocks (depth): 6
        # 2. Output channels: 80
        # 3. Group size: 16 => split channels into 5 groups
        # Input shape: (n, 32, 24, 24)
        # Output shape: (n, 80, 12, 12)
        nn.SequentialCell([
            ResNeXtBlock(32, 80, stride=2, group=5),
            ResNeXtBlock(80, 80, group=5),
            ResNeXtBlock(80, 80, group=5),
            ResNeXtBlock(80, 80, group=5),
            ResNeXtBlock(80, 80, group=5),
            ResNeXtBlock(80, 80, group=5)
        ]),

        # Head - global average pooling followed by linear layer
        # Input shape: (n, 80, 12, 12)
        # Output shape: (n, 10)
        nn.SequentialCell([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dense(80, 10)
        ])
    ])

    net_amp = amp.auto_mixed_precision(network=net, amp_level='O2')
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = nn.SGD(params=net_amp.trainable_params(), learning_rate=learning_rate)

    model = Model(
        network=net_amp,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={'accuracy', 'loss'}
    )
    callbacks = [
        LossMonitor(per_print_times=10),
        MLflowFashionMNISTCallback(
            run_name='01-designing-convolution-network-architectures',
            network_type='regnetx',
            loss_fn='softmax_cross_entropy',
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

if __name__ == '__main__':
    main()
