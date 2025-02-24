"""Combines PurpleAir downloaded data into usable training data."""

from absl import app, flags

import dataclasses
import datetime
import json
import logging
import pathlib
import random

_HISTORY_LENGTH = flags.DEFINE_integer(
  'history_length', 7,
  'How many days into the past should the history run? A longer history gives '
  'more training data but fewere datapoints to train with.')

_PREDICTION_LENGTH = flags.DEFINE_integer(
  'prediction_length', 2,
  'How many days into the future should we try collect outcomes?'
)

_SCRAPER_RESOLUTION = flags.DEFINE_integer(
  'scraper_resolution', 4, 'How many hours a single dataset entry represents. '
  'See purpleair_scraper.py'
)

_TRAIN_SPLIT = flags.DEFINE_float(
  'train_split', 0.85, 'The split between training and validation data.')


@dataclasses.dataclass
class Measurement:
  timestamp: datetime.datetime
  # Relative humidity.
  humidity: float
  # Temperature in Celsius.
  temperature: float
  # Pressure in Millibars.
  pressure: float
  
  @classmethod
  def get_csv_header(cls) -> str:
    return 'timestamp,humidity,temperature,pressure'
  
  def get_csv_line(self) -> str:
    return f'{self.timestamp.isoformat()},{self.humidity},{self.temperature},{self.pressure}'
  
  def get_history_start(self) -> datetime.datetime:
    return self.timestamp - datetime.timedelta(days=_HISTORY_LENGTH.value)
  
  def get_prediction_end(self) -> datetime.datetime:
    return self.timestamp + datetime.timedelta(days=_PREDICTION_LENGTH.value)


@dataclasses.dataclass
class TrainingSet:
  """A single line in the training set. Contains the history, current and future Measurements."""
  # The current measurement.
  actual: Measurement
  # The history, defined by --history_length, of measurments leading to this.
  history: list[Measurement]
  # Future measurements, defined by --prediction_length.
  future: list[Measurement]
  
  def is_valid(self) -> bool:
    expected_history = _SCRAPER_RESOLUTION.value * _HISTORY_LENGTH.value
    expected_prediction = _SCRAPER_RESOLUTION.value * _PREDICTION_LENGTH.value
        
    return len(self.history) == expected_history and len(self.future) == expected_prediction
    
  def __str__(self):
    return f',{self.actual.timestamp.isoformat()}; hist {len(self.history)}; pred {len(self.future)}'


# see https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
class EnhancedJSONEncoder(json.JSONEncoder):
  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    if isinstance(o, datetime.datetime):
      return o.isoformat()
    return super().default(o)


def _fahrenheit_to_celsius(f: float) -> float:
  # See https://api.purpleair.com/#api-sensors-get-sensors-data; the
  # internal temperature is ~8F higher than ambient.
  return (f - 32.0 - 8.0) / 1.8


def _calibrate_humidity(h: float) -> float:
  # See https://api.purpleair.com/#api-sensors-get-sensors-data; the
  # humity is ~4% lower than ambient conditions.
  return h + 4.0


def _write_csv(data: list[Measurement] | list[TrainingSet], 
               outfile: pathlib.Path) -> None:
  logging.info('Writing %d entries to %s', len(data), outfile)
  with open(outfile, 'w') as f:
    f.write(data[0].get_csv_header() + '\n')
    for d in data:
      f.write(d.get_csv_line() + '\n')


def _get_time_bracketed_entries(start: datetime.datetime, end:datetime.datetime,
                                data: list[Measurement]):
  """Returns data entries between start and end (inclusive)."""
  entries = []
  for d in data:
    if d.timestamp >= start and d.timestamp <= end:
      entries.append(d) 
  return entries  


def _collect_training_data(data: list[Measurement]) -> list[TrainingSet]:
  training_set = []
  
  # Small epsilon to not include the current measurement when querying for past
  # and future entries. 
  epsilon = datetime.timedelta(seconds=1)
  
  for measurement in data:
    t = TrainingSet(
      actual=measurement,
      history=_get_time_bracketed_entries(
        measurement.get_history_start(), 
        measurement.timestamp - epsilon, data),
      future=_get_time_bracketed_entries(
        measurement.timestamp + epsilon, 
        measurement.get_prediction_end(), data)
    )
    
    if t.is_valid():    
      training_set.append(t)
    else:
      logging.warning('Training set invalid: %s', t)
  return training_set


def main(argv: list[str]) -> None:
  if len(argv) != 2:
    raise app.UsageError(f'Invalid args; usage: {argv[0]} <input file> [optional flags]')
  
  input_file = pathlib.Path(argv[1])
  if not input_file.exists():
    raise FileNotFoundError(input_file)

  with open(input_file, 'r') as f:
    logging.info('Reading from %s', input_file)
    data = json.load(f)
  
  start = datetime.datetime.fromtimestamp(data['start_timestamp'])
  end = datetime.datetime.fromtimestamp(data['end_timestamp'])
  fields = data['fields']
  
  logging.info('Read data from %s to %s, fields: %s', start, end, fields)

  cleaned_data = []
  for d in data['data']:
    cleaned_data.append(
      Measurement(
        timestamp=datetime.datetime.fromtimestamp(d[0]),
        humidity=_calibrate_humidity(d[1]),
        temperature=_fahrenheit_to_celsius(d[2]),
        pressure=d[3]        
      ))
    
  logging.info('Sorting %d entries', len(cleaned_data))
  cleaned_data = sorted(cleaned_data, key=lambda d: d.timestamp)

  raw_csv = input_file.with_suffix('.csv')
  _write_csv(cleaned_data, raw_csv)
  
  training_data = _collect_training_data(cleaned_data)
  logging.info('Collected %d entries', len(training_data))
  
  random.shuffle(training_data)
  
  training_idx = int(len(training_data) * _TRAIN_SPLIT.value)  
  validation_data = training_data[training_idx:]
  training_data = training_data[:training_idx]
    
  logging.info('Created training data set with %d entries and validation data set with %d entries', len(training_data), len(validation_data))
  
  with open(input_file.with_suffix('.validation.json'), 'w') as file:
    logging.info('Writing validation data to %s', file.name)
    file.write(json.dumps(validation_data, indent=2, cls=EnhancedJSONEncoder))
    
  with open(input_file.with_suffix('.training.json'), 'w') as file:
    logging.info('Writing training data to %s', file.name)
    file.write(json.dumps(training_data, indent=2, cls=EnhancedJSONEncoder))


if __name__ == '__main__':
  app.run(main)