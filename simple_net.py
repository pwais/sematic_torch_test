# Copyright 2023 Maintainers of sematic_torch_test

"""A simple Pytorch nerual-net-based classifier and training / testing dataset
generator.  The code in this module is plain pytorch; all Sematic integration
happens in sematic_pipeline.py.

Things to notice:
 * We organize system parameters into dataclasses because the Sematic server
    has built-in support for visualizing and tracing dataclasses.
 * For now we do not save model weights anywhere; for that you want your
    cluster to have some sort of persistent storage.
 * When running this demo code locally via e.g.
      `python main.py --run-torch-only`
    or
      `sematic run main.py`
    you can add a `breakpoint()` anywhere in this code to debug.  When running
    with cloud execution, though, you cannot use `breakpoint()` / debugging
    tools, you need to rely on the data / metrics / artifacts etc tracked
    through Sematic and persisted to cluster storage.

See also Sematic's built-in MNIST example, which is a bit more complicated:
 * https://docs.sematic.dev/onboarding/real-example
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data as data
from loguru import logger as log
from tqdm import tqdm


### Utils

def get_device_info(device: torch.device) -> str:
  """Helps report info about the (cloud) GPU used in the Sematic dashboard.
  Simply a debugging aid."""
  
  if device.type != 'cuda':
    return f'Using non-cuda device {device}'

  if not torch.cuda.is_available():
    return 'Torch cuda unavailable, got device {device}'

  gpus_visible = torch.cuda.device_count()
  gpu_name = torch.cuda.get_device_name(device)
  return f'Using GPU {gpu_name}, num GPUs visible: {gpus_visible}'

def get_hostname() -> str:
  import socket
  return socket.gethostname()


### Model


@dataclass
class FeedForwardNetParams:
  num_inputs: int = 2
  num_hidden: int = 4
  num_outputs: int = 1


class FeedForwardNet(nn.Module):

  def __init__(self, params: FeedForwardNetParams):
    super().__init__()
    self.params = params
    self.layer1 = nn.Linear(self.params.num_inputs, self.params.num_hidden)
    self.act_fn = nn.Sigmoid()
    self.layer2 = nn.Linear(self.params.num_hidden, self.params.num_outputs)

  def forward(self, x):
    x = self.layer1(x)
    x = self.act_fn(x)
    x = self.layer2(x)
    return x


### Data


@dataclass
class NoisyXORDatasetParams:
  size: int = 1000
  noise_std: float = 0.1


class NoisyXORDataset(data.Dataset):

  def __init__(self, params: NoisyXORDatasetParams):
    super().__init__()
    self.params = params
    self._precompute()

  def _precompute(self):
    data = torch.randint(low=0,
                         high=2,
                         size=(self.params.size, 2),
                         dtype=torch.float32)
    label = (data.sum(dim=1) == 1).to(torch.long)
    data += self.params.noise_std * torch.randn(data.shape)

    self.data = data
    self.label = label

  def __len__(self):
    return self.params.size

  def __getitem__(self, idx):
    data_point = self.data[idx]
    data_label = self.label[idx]
    return data_point, data_label


### Training Harness


@dataclass
class TrainTestParams:
  num_epochs: int = 1000
  batch_size: int = 100
  learning_rate: float = 0.02
  momentum: float = 0.9

  seed: int = 1337
  use_cuda: bool = True

  test_batch_size: int = 100

  train_dataset_params: NoisyXORDatasetParams = NoisyXORDatasetParams()
  test_dataset_params: NoisyXORDatasetParams = NoisyXORDatasetParams()
  model_params: FeedForwardNetParams = FeedForwardNetParams()


@dataclass
class TrainTestResults:
  train_params: TrainTestParams = TrainTestParams()

  final_train_loss: float = 0.
  accuracy: float = 0.

  device_info: str = ''
  training_hostname: str = ''


def train_test_model(params: TrainTestParams) -> TrainTestResults:

  log.info(f"Beginning train/test of model with params {params}")

  torch.manual_seed(params.seed)

  train_dataset = NoisyXORDataset(params.train_dataset_params)
  train = data.DataLoader(train_dataset,
                          batch_size=params.batch_size,
                          shuffle=True)

  device = torch.device('cuda' if params.use_cuda else 'cpu')
  log.info(f"... using device {device} ...")

  model = FeedForwardNet(params.model_params)
  model.train()
  model = model.to(device)
  log.info(f"... torch model: \n{model}\n ...")

  compute_loss = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(model.parameters(),
                              lr=params.learning_rate,
                              momentum=params.momentum)

  log.info(f"... training for {params.num_epochs} epochs ...")
  for epoch in tqdm(range(params.num_epochs), desc='Training'):
    for xs, labels in train:
      xs = xs.to(device)
      labels = labels.to(device)

      ys = model(xs)
      loss = compute_loss(ys.squeeze(dim=1), labels.float())

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()

    if epoch % 100 == 0:
      log.info(f"Epoch: {epoch} Loss: {loss}")
  log.info(f"... final loss {loss} ...")

  test_dataset = NoisyXORDataset(params.test_dataset_params)
  test = data.DataLoader(test_dataset,
                         batch_size=params.test_batch_size,
                         shuffle=False,
                         drop_last=False)

  model.eval()
  tps = 0.
  n = 0.

  log.info(f"... testing on {len(test_dataset)} examples ...")
  with torch.no_grad():
    for xs, labels in tqdm(test, desc='Testing'):
      xs = xs.to(device)
      labels = labels.to(device)

      ys = model(xs)
      preds = torch.sigmoid(ys.squeeze())

      tps += ((preds >= 0.5).long() == labels).sum()
      n += len(labels)
  accuracy = 100. * tps / n
  log.info(f"... test accuracy {accuracy:4.2f}% ...")

  results = TrainTestResults(
    train_params=params,

    # NB: need to unpack torch tensor types because they are not
    # Sematic-serializable
    final_train_loss=loss.item(),
    accuracy=accuracy.item(),
    device_info=get_device_info(device),
    training_hostname=get_hostname(),
  )

  return results
