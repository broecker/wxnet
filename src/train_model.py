from absl import app, flags

import json
import logging
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

_TRAINING_DATA = flags.DEFINE_string('training_data', None, 'Path to training data json file.', required=True)


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Too many parameters; usage: {argv[0]}')
  
  training_data = pathlib.Path(_TRAINING_DATA.value)
  if not training_data.exists():
    raise FileNotFoundError(training_data)
  
  with open(training_data, 'r') as file:
    logging.info('Reading traning data from %s ... ', training_data)
    training_data = json.load(file)
    
  logging.info('Read %d entries.', len(training_data))
  if not training_data:
    raise RuntimeError('No data found.')
  
  

if __name__ == '__main__':
  app.run(main)