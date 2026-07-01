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
00-garment-classifier-mindspore.py
Adaptation of "Training with PyTorch" official tutorial with MindSpore
https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""

def transform_ds(dataset):
    image_transforms = [
        vision.Resize(size=(28, 28)),
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

class MLflowFashionMNISTCallback(Callback):
    def __init__(self):
        super().__init__()

        run_name = '00-garment-classifier-mindspore'
        self.run = mlflow.start_run(run_name=run_name)

        hyperparameters = {
            'learning_rate': 0.1,
            'weight_decay': 0,
            'momentum': 0,
            'loss': 'softmax_cross_entropy',
            'optimizer': 'sgd',
            'batch_size': 128,
            'network_type': 'lenet',
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
    experiment_name = '00-sanity-check'
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
        nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Dense(256, 120, activation='relu'),
        nn.Dense(120, 84, activation='relu'),
        nn.Dense(84, 10)
    ])

    net_amp = amp.auto_mixed_precision(network=net, amp_level='O2')
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    optimizer = nn.SGD(params=net_amp.trainable_params(), learning_rate=0.1)

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
