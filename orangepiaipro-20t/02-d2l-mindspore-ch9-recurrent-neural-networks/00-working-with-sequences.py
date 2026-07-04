import matplotlib.pyplot as plt
import mindspore
import mindspore.amp as amp
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.nn as nn
import mindspore.ops as ops
import mlflow
import os

from mindspore import dtype as mstype
from mindspore.train import Callback, LossMonitor, Model

"""
00-working-with-sequences.py
MindSpore adaptation of D2L chapter 9.1
https://d2l.ai/chapter_recurrent-neural-networks/sequence.html
"""

def transform_ds(dataset, batch_size):
    feature_transforms = [
        transforms.TypeCast(data_type=mstype.float32)
    ]
    label_transforms = [
        transforms.TypeCast(data_type=mstype.float32)
    ]
    dataset = dataset.map(operations=feature_transforms, input_columns='feature')
    dataset = dataset.map(operations=label_transforms, input_columns='label')
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    return dataset

def k_step_pred(k, x, model, T=1000, tau=4):
    features = []
    for i in range(tau):
        features.append(x[i:i+T-tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model.predict(ops.stack(features[i:i+tau], axis=1))
        features.append(ops.reshape(preds, (-1,)))
    return features[tau:]

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
        mlflow.log_metric('train_loss', current_loss, step=cb_params.cur_step_num)

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if hasattr(cb_params, 'eval_results') and cb_params.eval_results:
            val_loss = cb_params.eval_results.get('loss', 0.0)
            mlflow.log_metric('val_loss', val_loss, step=cb_params.cur_epoch_num)

    def on_train_end(self, run_context):
        mlflow.end_run()

def main():
    mindspore.set_device(device_target='Ascend', device_id=0)

    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    MPLBACKEND = os.getenv('MPLBACKEND')
    print(f'Using MLflow tracking URI: {MLFLOW_TRACKING_URI}')
    print(f'Using Matplotlib backend: {MPLBACKEND}')

    experiment_name = '02-d2l-mindspore-ch9-recurrent-neural-networks'
    experiment = mlflow.set_experiment(experiment_name=experiment_name)

    """
    Generate some sample data using sine function with additive noise
    """
    T = 1000
    t = 1 + ops.arange(T)
    x = ops.sin(0.01 * t) + ops.randn(T) * 0.2

    """
    Plot the sample data and save as PNG image
    """
    fig = plt.figure(figsize=(8, 5))
    plt.plot(t.asnumpy(), x.asnumpy(), label='$x = \\sin(0.01 t) + \\epsilon$', linestyle='-')
    plt.title('Sine function with additive noise')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(
        '00-ch9-sequence-synthetic-data.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

    """
    Prepare the sequence data and train our model
    Given the past `tau` samples of our data, we want to predict the next data point
    Let's use a simple linear regression model with `tau` features and 1 label
    """
    learning_rate = 0.01
    batch_size = 16
    epochs = 5
    tau = 4 # Assume data satisfies 4-th order Markov condition
    num_train = 628 # Take 1 full period of sine wave for our training set
    features = ops.stack([x[i:T-tau+i] for i in range(tau)], axis=1)
    labels = ops.reshape(x[tau:], (-1, 1))
    X_train = features[:num_train].asnumpy()
    y_train = labels[:num_train].asnumpy()
    X_test = features[num_train:].asnumpy()
    y_test = labels[num_train:].asnumpy()
    train_ds = ds.NumpySlicesDataset(data=(X_train, y_train), column_names=['feature', 'label'], shuffle=True)
    test_ds = ds.NumpySlicesDataset(data=(X_test, y_test), column_names=['feature', 'label'], shuffle=True)
    train_ds = transform_ds(train_ds, batch_size=batch_size)
    test_ds = transform_ds(test_ds, batch_size=batch_size)
    net = nn.Dense(tau, 1)
    net_amp = amp.auto_mixed_precision(network=net, amp_level='O2')
    loss_fn = nn.MSELoss()
    optimizer = nn.SGD(params=net_amp.trainable_params(), learning_rate=learning_rate)
    model = Model(
        network=net_amp,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={'loss'}
    )
    callbacks = [
        LossMonitor(per_print_times=10),
        MLflowLogging(
            run_name='00-working-with-sequences',
            network_type='linear',
            loss_fn='mse',
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
    Visualize one-step-ahead predictions and see how well our model performs
    """
    onestep_preds = model.predict(features)
    fig = plt.figure(figsize=(8, 5))
    t_tau = t[tau:].asnumpy()
    plt.plot(t_tau, ops.squeeze(labels, axis=1).asnumpy(), label='labels', linestyle='-')
    plt.plot(t_tau, ops.squeeze(onestep_preds, axis=1).asnumpy(), label='1-step preds', linestyle='--')
    plt.title('One-step-ahead predictions')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(
        '01-ch9-sequence-onestep-preds.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

    """
    Visualize multistep-ahead predictions and see how well our model performs
    """
    multistep_preds = ops.deepcopy(x)
    for i in range(num_train + tau, T):
        multistep_preds[i] = model.predict(ops.reshape(multistep_preds[i-tau:i], (1, -1)))
    fig = plt.figure(figsize=(8, 5))
    t_tau = t[tau:].asnumpy()
    t_num_train_tau = t[num_train+tau:].asnumpy()
    plt.plot(t_tau, ops.squeeze(onestep_preds, axis=1).asnumpy(), label='1-step preds', linestyle='-')
    plt.plot(t_num_train_tau, multistep_preds[num_train+tau:].asnumpy(), label='multistep preds', linestyle='--')
    plt.title('Multistep-ahead predictions')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(
        '02-ch9-sequence-multistep-preds.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

    """
    Visualize k-step-ahead predictions and see how well our model performs
    k = 1, 4, 16, 64
    """
    k = 64
    preds = k_step_pred(k=k, x=x, model=model, T=T, tau=tau)
    fig = plt.figure(figsize=(8, 5))
    t_tau_k = t[tau+k-1:].asnumpy()
    steps = (
        (1, '-'),
        (4, '--'),
        (16, '-.'),
        (64, ':')
    )
    for step, style in steps:
        plt.plot(t_tau_k, preds[step - 1].asnumpy(), label=f'{step}-step preds', linestyle=style)
    plt.title('k-step-ahead predictions')
    plt.xlabel('time')
    plt.ylabel('x')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(
        '03-ch9-sequence-k-step-preds.png',
        dpi=300,
        bbox_inches='tight',
        transparent=False,
        format='png'
    )
    plt.close(fig)

if __name__ == '__main__':
    main()
