"""Combines PurpleAir downloaded data into usable training data."""

from absl import app, flags

import dataclasses
import datetime
import json
import logging
import pathlib


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
  input_file = pathlib.Path(argv[1])
  output_file = pathlib.Path(argv[2])
  if not input_file.exists():
    raise FileNotFoundError(input_file)

  logging.info('Reading from %s, writing to %s', input_file, output_file)
  
  with open(input_file, 'r') as f:
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

  _write_csv(cleaned_data, output_file)

  


if __name__ == '__main__':
  app.run(main)