from datetime import datetime, timedelta
import logging 

logger = logging.getLogger(__name__)


def load_forecasts(conf):
    if "type" in conf["predict"]["forecasts"]:
        type = conf["predict"]["forecasts"]
        if type == "10day_year":
            return generate_OneYearDaily_10day_forecast(conf["predict"]["forecasts"]["start_date"])
        else:
            logger.warning(f"Forecast type {type} not supported")
            raise OSError
    else:
        return conf["predict"]["forecasts"]


# Function to generate 10-day forecast given a start date
def generate_OneYearDaily_10day_forecast(start_date):
    forecasts = []
    current_date = start_date

    # Generate forecast for each day
    for _ in range(365):
        end_date = current_date + timedelta(days=9, hours=23)  # Set end time to 23:00:00 on the 10th day
        forecasts.append([current_date.strftime("%Y-%m-%d %H:%M:%S"),
                          end_date.strftime("%Y-%m-%d %H:%M:%S")])
        current_date += timedelta(days=1)

    return forecasts

if __name__ == "__main__":
    # Generate forecasts for each day of the year
    start_date = datetime(2020, 1, 1)
    forecasts = generate_forecast(start_date)
    
    # Print example forecasts
    for forecast in forecasts:
        print(forecast)
