#!/bin/zsh
# Save the DK1 onshore day-ahead forecast (today and tomorrow) once the given UTC time HHMM is reached.
# Run from the repository root: zsh src/snapshot_dayahead.sh 0800
target="$1"
while [ "$(date -u +%H%M)" -lt "$target" ]; do sleep 60; done
d0=$(date -u +%Y-%m-%d); d2=$(date -u -v+2d +%Y-%m-%d)
curl -s -m 60 "https://api.energidataservice.dk/dataset/Forecasts_Hour?start=${d0}T00:00&end=${d2}T00:00&filter=%7B%22PriceArea%22:%5B%22DK1%22%5D,%22ForecastType%22:%5B%22Onshore%20Wind%22%5D%7D&timezone=UTC&sort=HourUTC%20asc" > "data/forecast_snapshots/dayahead_snapshot_$(date -u +%Y%m%dT%H%M).json"
echo done
