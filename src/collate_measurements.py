"""Combines PurpleAir downloaded data into usable training data."""

from absl import app, flags

import dataclasses
import datetime
import json
import logging
import pathlib

_HISTORY_LENGTH = flags.DEFINE_integer(
  'history_length', 7,
  'How many days into the past should the history run? A longer history gives '
  'more training data but fewere datapoints to train with.')

_PREDICTION_LENGTH = flags.DEFINE_integer(
  'prediction_length', 2,
  'How many days into the future should we try collect outcomes?'
)


@dataclasses.dataclass
class Measurement:
  timestamp: datetime.datetime
  # Relative humidity.
  humidity: float
  # Temperature in Celsius.
  temperature: float
  # Pressure in Millibars.
  pressure: float
  
  def get_csv_line(self) -> str:
    return f'{self.timestamp.isoformat()},{self.humidity},{self.temperature},{self.pressure}\n'

@dataclasses.dataclass
class TrainingSet:
  """A single line in the training set. Contains the history, current and future Measurements."""
  # The current measurement.
  actual: Measurement
  # The history, defined by --history_length, of measurments leading to this.
  history: list[Measurement]
  # Future measurements, defined by --prediction_length.
  future: list[Measurement]


def _fahrenheit_to_celsius(f: float) -> float:
  # See https://api.purpleair.com/#api-sensors-get-sensors-data; the
  # internal temperature is ~8F higher than ambient.
  return (f - 32.0 - 8.0) / 1.8


def _calibrate_humidity(h: float) -> float:
  # See https://api.purpleair.com/#api-sensors-get-sensors-data; the
  # humity is ~4% lower than ambient conditions.
  return h + 4.0


def _write_csv(data: list[Measurement], outfile: pathlib.Path) -> None:
  logging.info('Writing %d entries to %s', len(data), outfile)
  with open(outfile, 'w') as f:
    f.write('timestamp,humidity,temperature,pressure\n')
    for d in data:
      f.write(d.get_csv_line())


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
  

if __name__ == '__main__':
  app.run(main)