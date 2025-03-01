"""Scrapes data from a single PurpleAir sensor to build models.

Dependencies:
  * absl-py
  * requests
"""

from absl import app, flags

import datetime
import json
import logging
import requests
import time


_PURPLEAIR_API_KEY = flags.DEFINE_string(
    "purpleair_key", None, "Purpleair API key.", required=True
)

_PURPLEAIR_STATION_ID = flags.DEFINE_integer(
    "purpleair_station", None, "Purpleair station to query.", required=True
)

_START_TIMESTAMP = flags.DEFINE_string(
    "start_timestamp",
    (datetime.datetime.now() - datetime.timedelta(days=14)).isoformat(),
    "The start date of the data export in ISO format. By default 2 weeks in the "
    "past.",
)

_END_TIMESTAMP = flags.DEFINE_string(
    "end_timestamp",
    datetime.datetime.now().isoformat(),
    "The end time of the data export in ISO format",
)

_OUTFILE = flags.DEFINE_string(
    "outfile",
    None,
    "Where we should write the data (as csv) to. If not set, we "
    "will create a new file in /tmp.",
)

_SLEEP_BETWEEN_REQUESTS = flags.DEFINE_integer(
    "sleep_between_requests", 5, "Wait time between requests in seconds."
)


def main(argv: list[str]) -> None:
    logging.info("Hello PurpleAir scraper!")
    logging.info(_PURPLEAIR_API_KEY.value)

    purpleair_headers = {
        "X-API-Key": _PURPLEAIR_API_KEY.value,
        "Content-Type": "application/json",
    }

    # Check the API key validity.
    response = requests.get(
        "https://api.purpleair.com/v1/keys", headers=purpleair_headers
    )
    if not response.ok:
        logging.error("Unable to query PurpleAir API: %s", response.reason)
        return

    if _OUTFILE.value:
        outfile = _OUTFILE.value
    else:
        outfile = f"/tmp/{_PURPLEAIR_STATION_ID.value}.json"
    logging.info("Will write to %s", outfile)

    # See https://community.purpleair.com/t/loop-api-calls-for-historical-data/4623
    # Query in 6 hour intervals.
    resolution = 360

    # Combine the response jsons into one giant object. The bulk is in the 'data'
    # array field.
    all_data = {}

    current_timestamp = datetime.datetime.fromisoformat(_START_TIMESTAMP.value)
    timedelta = datetime.timedelta(weeks=4)
    end_timestamp = datetime.datetime.fromisoformat(_END_TIMESTAMP.value)

    while current_timestamp < end_timestamp:
        end = current_timestamp + timedelta
        logging.info(
            "Querying station %d from %s to %s",
            _PURPLEAIR_STATION_ID.value,
            current_timestamp,
            end,
        )
        url = f"https://api.purpleair.com/v1/sensors/{_PURPLEAIR_STATION_ID.value}/history?start_timestamp={current_timestamp.timestamp()}&end_timestamp={end.timestamp()}&fields=humidity%2Cpressure%2Ctemperature&average={resolution}"

        response = requests.get(url, headers=purpleair_headers)
        current_timestamp += timedelta

        if not response.ok:
            logging.error("Unable to query PurpleAir API: %s", response.reason)
            response.raise_for_status()
        else:
            data = json.loads(response.text)
            if not all_data:
                logging.info("Setting initial data.")
                all_data = data
            else:
                read_end = datetime.datetime.fromtimestamp(data["end_timestamp"])
                logging.info("Appending data until %s", read_end)
                try:
                    all_data["data"].extend(data["data"])
                except KeyError:
                    logging.exception("Unable to find data array.")

        time.sleep(_SLEEP_BETWEEN_REQUESTS.value)

    logging.info("Writing %s ...", outfile)
    with open(outfile, "w") as f:
        f.write(json.dumps(all_data, indent=2))


if __name__ == "__main__":
    app.run(main)
