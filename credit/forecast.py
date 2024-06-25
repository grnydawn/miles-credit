from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_forecasts(conf):
    if "type" in conf["predict"]["forecasts"]:
        forecast_type = conf["predict"]["forecasts"]["type"]
        start_date = datetime(
            conf["predict"]["forecasts"]["start_year"],
            conf["predict"]["forecasts"]["start_month"],
            conf["predict"]["forecasts"]["start_day"]
        )
        start_hours = conf["predict"]["forecasts"].get("start_hours", [0])  # Default to [0] if not provided
        if forecast_type == "10day_year":
            return generate_forecasts(start_date, days=10, duration=365, start_hours=start_hours)
        elif forecast_type == "custom":
            days = conf["predict"]["forecasts"].get("days", 10)
            duration = conf["predict"]["forecasts"].get("duration", 365)
            return generate_forecasts(start_date, days=days, duration=duration, start_hours=start_hours)
        else:
            logger.warning(f"Forecast type '{forecast_type}' not supported")
            raise ValueError(f"Forecast type '{forecast_type}' not supported")
    else:
        return conf["predict"]["forecasts"]


# Function to generate forecasts for specified duration
def generate_forecasts(start_date, days=10, duration=365, start_hours=[0]):
    forecasts = []
    current_date = start_date  # Use the provided start_date directly

    # Generate forecast for each day
    for _ in range(duration):
        for hour in start_hours:
            start_datetime = current_date + timedelta(hours=hour)
            end_datetime = start_datetime + timedelta(days=days-1, hours=23)
            forecasts.append([start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                              end_datetime.strftime("%Y-%m-%d %H:%M:%S")])
        current_date += timedelta(days=1)

    return forecasts


if __name__ == "__main__":
    # Generate forecasts for each day of the year
    # start_date = datetime(2020, 1, 1)
    # forecasts = generate_forecasts(start_date)

    config_file = "/glade/derecho/scratch/schreck/repos/miles-credit/results/wxformer/quarter/zscore_multi/model.yml"

    import yaml
    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    forecasts = load_forecasts(conf)
    # Print example forecasts
    for forecast in forecasts:
        print(forecast)
