from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Function to load forecasts based on configuration
def load_forecasts(conf):
    if "type" in conf["predict"]["forecasts"]:
        forecast_type = conf["predict"]["forecasts"]["type"]
        if forecast_type == "10day_year":
            start_date = conf["predict"]["forecasts"]["start_date"]
            return generate_forecasts(start_date, days=10, duration=365)
        elif forecast_type == "custom":
            start_date = conf["predict"]["forecasts"]["start_date"]
            days = conf["predict"]["forecasts"].get("days", 10)
            duration = conf["predict"]["forecasts"].get("duration", 365)
            return generate_forecasts(start_date, days=days, duration=duration)
        else:
            logger.warning(f"Forecast type '{forecast_type}' not supported")
            raise ValueError(f"Forecast type '{forecast_type}' not supported")
    else:
        return conf["predict"]["forecasts"]


# Function to generate forecasts for specified duration
def generate_forecasts(start_date, days=10, duration=365):
    forecasts = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    # Generate forecast for each day
    for _ in range(duration):
        end_date = current_date + timedelta(days=days-1, hours=23)  # Set end time to 23:00:00 on the last day
        forecasts.append([current_date.strftime("%Y-%m-%d %H:%M:%S"),
                          end_date.strftime("%Y-%m-%d %H:%M:%S")])
        current_date += timedelta(days=1)

    return forecasts


if __name__ == "__main__":
    # Generate forecasts for each day of the year
    start_date = datetime(2020, 1, 1)
    forecasts = generate_forecasts(start_date)
    
    # Print example forecasts
    for forecast in forecasts:
        print(forecast)
