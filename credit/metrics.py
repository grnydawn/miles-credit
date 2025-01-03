import torch
import numpy as np
from datetime import datetime
from credit.loss import latitude_weights


class LatWeightedMetrics:
    def __init__(self, conf):
        self.conf = conf
        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = (
            conf["model"]["levels"]
            if "levels" in conf["model"]
            else conf["model"]["frames"]
        )

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # forecast_datetime is passed for interface consistency but not used here

        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = (
            self.w_lat.to(dtype=pred.dtype, device=pred.device)
            if self.w_lat is not None
            else 1.0
        )
        w_var = (
            self.w_var.to(dtype=pred.dtype, device=pred.device)
            if self.w_var is not None
            else 1.0
        )

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            error = pred - y
            for i, var in enumerate(self.vars):
                pred_prime = pred[:, i] - torch.mean(pred[:, i])
                y_prime = y[:, i] - torch.mean(y[:, i])

                # Add epsilon to avoid division by zero
                epsilon = 1e-7

                denominator = (
                    torch.sqrt(
                        torch.sum(w_var * w_lat * pred_prime**2)
                        * torch.sum(w_var * w_lat * y_prime**2)
                    )
                    + epsilon
                )

                loss_dict[f"acc_{var}"] = (
                    torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                )
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(
                        torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1))
                    )
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (
                    torch.abs(error[:, i]) * w_lat * w_var
                ).mean()

        # Calculate metrics averages
        loss_dict["acc"] = np.mean(
            [loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k]
        )
        loss_dict["rmse"] = np.mean(
            [loss_dict[k].cpu() for k in loss_dict.keys() if "rmse_" in k]
        )
        loss_dict["mse"] = np.mean(
            [
                loss_dict[k].cpu()
                for k in loss_dict.keys()
                if "mse_" in k and "rmse_" not in k
            ]
        )
        loss_dict["mae"] = np.mean(
            [loss_dict[k].cpu() for k in loss_dict.keys() if "mae_" in k]
        )

        return loss_dict


class LatWeightedMetricsClimatology:
    def __init__(self, conf, climatology=None):
        self.conf = conf
        self.climatology = climatology  # xarray Dataset with climatology data

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = (
            conf["model"]["levels"]
            if "levels" in conf["model"]
            else conf["model"]["frames"]
        )

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]

        self.vars += surface_vars
        self.vars += diag_vars
        self.acc_vars = surface_vars + diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None

    def get_climatology(self, forecast_datetime, variable):
        """Extract the climatology for the given forecast datetime and variable."""
        forecast_datetime = datetime.utcfromtimestamp(forecast_datetime)
        dayofyear = forecast_datetime.timetuple().tm_yday
        hour = forecast_datetime.hour

        # Extract climatology slice from xarray dataset
        climatology_slice = self.climatology[variable].sel(
            dayofyear=dayofyear, hour=hour, method="nearest"
        )
        # Convert to PyTorch tensor
        return torch.tensor(climatology_slice.values, dtype=torch.float32)

    def __call__(self, pred, y, transform=None, forecast_datetime=None):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = (
            self.w_lat.to(dtype=pred.dtype, device=pred.device)
            if self.w_lat is not None
            else 1.0
        )
        w_var = (
            self.w_var.to(dtype=pred.dtype, device=pred.device)
            if self.w_var is not None
            else 1.0
        )

        loss_dict = {}
        with torch.no_grad():
            # Compute ACC for acc_vars using anomalies
            anamoly_scores = False
            if self.climatology and forecast_datetime:
                anomalies_pred = []
                anomalies_y = []
                acc_pred = pred
                acc_y = y

                # Get the list of variables from the climatology file
                clim_vars = list(self.climatology.data_vars)

                # Ensure self.acc_vars is in the same order as clim_vars
                ordered_acc_vars = [var for var in clim_vars if var in self.vars]

                # Reorder acc_pred and acc_y to match ordered_acc_vars
                indices = [self.acc_vars.index(var) for var in ordered_acc_vars]
                acc_pred = acc_pred[:, indices]
                acc_y = acc_y[:, indices]

                # Compute anamolies
                for i, var in enumerate(ordered_acc_vars):
                    clim = self.get_climatology(forecast_datetime, var).to(
                        dtype=pred.dtype, device=pred.device
                    ).unsqueeze(0)
                    anomalies_pred.append(acc_pred[:, i] - clim)
                    anomalies_y.append(acc_y[:, i] - clim)

                anomalies_pred = torch.stack(anomalies_pred, dim=1)
                anomalies_y = torch.stack(anomalies_y, dim=1)

                for i, var in enumerate(self.acc_vars):
                    pred_prime = anomalies_pred[:, i] - torch.mean(anomalies_pred[:, i])
                    y_prime = anomalies_y[:, i] - torch.mean(anomalies_y[:, i])

                    # Add epsilon to avoid division by zero
                    epsilon = 1e-7
                    denominator = (
                        torch.sqrt(
                            torch.sum(w_var * w_lat * pred_prime**2)
                            * torch.sum(w_var * w_lat * y_prime**2)
                        )
                        + epsilon
                    )
                    loss_dict[f"acc_{var}"] = (
                        torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
                    )
                anamoly_scores = True

            # Compute RMSE, MSE, MAE for all vars
            error = pred - y
            for i, var in enumerate(self.vars):
                loss_dict[f"rmse_{var}"] = torch.mean(
                    torch.sqrt(
                        torch.mean(error[:, i] ** 2 * w_lat * w_var, dim=(-2, -1))
                    )
                )
                loss_dict[f"mse_{var}"] = (error[:, i] ** 2 * w_lat * w_var).mean()
                loss_dict[f"mae_{var}"] = (
                    torch.abs(error[:, i]) * w_lat * w_var
                ).mean()

            # Compute average metrics
            if anamoly_scores:
                loss_dict["acc"] = np.mean(
                        [loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k]
                )
            loss_dict["rmse"] = np.mean(
                [loss_dict[k].cpu().item() for k in loss_dict.keys() if "rmse_" in k]
            )
            loss_dict["mse"] = np.mean(
                [loss_dict[k].cpu().item() for k in loss_dict.keys() if "mse_" in k]
            )
            loss_dict["mae"] = np.mean(
                [loss_dict[k].cpu().item() for k in loss_dict.keys() if "mae_" in k]
            )

        return loss_dict


if __name__ == "__main__":
    import yaml
    import logging
    import xarray as xr
    from credit.parser import credit_main_parser

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Open an example config
    with open(
        "/glade/derecho/scratch/schreck/repos/miles-credit/production/multistep/wxformer_6h/model.yml"
    ) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(
        conf, parse_training=True, parse_predict=False, print_summary=False
    )

    # Climatology file
    fn = '/glade/campaign/cisl/aiml/ksha/CREDIT_arXiv/VERIF/verif_6h/ERA5_clim/ERA5_clim_1990_2019_6h_interp.nc'
    climatology_data = xr.open_dataset(fn)

    # Make some fake data

    true = torch.tensor(np.random.rand(1, 71, 640, 1280), dtype=torch.float32)
    pred = torch.tensor(np.random.rand(1, 71, 640, 1280), dtype=torch.float32)

    logger.info("Computing metrics. ACC without a climatology")

    # Initialize the metrics class with the climatology data
    metrics = LatWeightedMetrics(conf=conf)

    # Compute metrics
    results = metrics(pred, true)

    # Display results
    for key, value in results.items():
        print(f"{key}: {value}")

    # Comptue metrics, and ACC correctly.

    logger.info("Computing metrics. ACC with a climatology")

    # Initialize the metrics class with the climatology data
    metrics = LatWeightedMetricsClimatology(conf=conf, climatology=climatology_data)

    # Define a forecast datetime (should align with the climatology dataset)
    forecast_datetime = datetime(2024, 6, 15, 12)  # Example forecast datetime

    # Compute metrics
    results = metrics(pred, true, forecast_datetime=forecast_datetime)

    # Display results
    for key, value in results.items():
        print(f"{key}: {value}")
