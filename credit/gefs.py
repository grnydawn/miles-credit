import xarray as xr
import numpy as np
import gcsfs
from tqdm import tqdm
import pandas as pd
from os.path import join
from scipy.sparse import csr_matrix
import logging


def download_gefs_run(init_date_str, out_path, n_pert_members=30):
    init_date = pd.Timestamp(init_date_str)
    init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
    bucket = "gs://gfs-ensemble-forecast-system/"
    members = ["c00"]
    if n_pert_members > 0:
        members.extend([f"p{m:02d}" for m in range(1, n_pert_members + 1)])
    ens_path = f"{bucket}{init_date_path}/atmos/init/"
    fs = gcsfs.GCSFileSystem(token="anon")
    logging.info(f"Downloading GEFS initialization for {init_date_str}")
    for member in tqdm(members):
        member_path = join(ens_path, member)
        out_member_path = join(out_path, init_date_path, member)
        if fs.exists(member_path):
            fs.get(member_path, out_member_path, recursive=True)
    return


def load_member_tiles(path: str, init_date_str: str, member: str, variables: str):
    num_tiles = 6
    init_date = pd.Timestamp(init_date_str)
    init_date_path = init_date.strftime("gefs.%Y%m%d/%H")
    out_member_path = join(path, init_date_path, member)
    member_tiles = []
    all_ua_variables = [
        "ps",
        "w",
        "zh",
        "t",
        "delp",
        "sphum",
        "liq_wat",
        "o3mr",
        "ice_wat",
        "rainwat",
        "snowwat",
        "graupel",
        "u_w",
        "v_w",
        "u_s",
        "v_s",
    ]
    all_surface_variables = [
        "ps",
        "w",
        "zh",
        "t",
        "delp",
        "sphum",
        "liq_wat",
        "o3mr",
        "ice_wat",
        "rainwat",
        "snowwat",
        "graupel",
        "u_w",
        "v_w",
        "u_s",
        "v_s",
    ]
    select_ua_variables = np.intersect1d(all_ua_variables, variables)
    select_surface_variables = np.intersect1d(all_surface_variables, variables)
    for t in range(1, num_tiles + 1):
        tile_ua_file = join(out_member_path, f"gfs_data.tile{t:02d}.nc")
        tile_sfc_file = join(out_member_path, f"sfc_data.tile{t:02d}.nc")
        if len(select_ua_variables) > 0:
            with xr.open_dataset(tile_ua_file) as ua_ds:
                member_tiles.append(ua_ds[select_ua_variables].load())
            if len(select_surface_variables) > 0:
                with xr.open_dataset(tile_sfc_file) as sfc_ds:
                    member_tiles[-1] = xr.merge(
                        [member_tiles[-1], sfc_ds[select_surface_variables].load()]
                    )
        elif len(select_surface_variables) > 0:
            with xr.open_dataset(tile_sfc_file) as sfc_ds:
                member_tiles.append(sfc_ds[select_surface_variables].load())
        else:
            raise ValueError("You did not request any valid GEFS variables.")
    return member_tiles


def unstagger_winds(ds, u_var="u_s", v_var="v_w", out_u="u_a", out_v="v_a"):
    ds["lev"] = np.arange(ds.sizes["lev"]).astype(np.float32)
    ds["lev"].attrs["axis"] = "Z"
    ds["x"] = np.arange(ds.sizes["lon"])
    ds["y"] = np.arange(ds.sizes["lon"])
    ds["x"].attrs["axis"] = "X"
    ds["y"].attrs["axis"] = "Y"
    ds[out_u] = xr.DataArray(
        0.5 * ds[u_var][:, :-1, :].values + ds[u_var][:, 1:, :].values,
        coords=dict(
            lev=ds["lev"],
            y=ds["y"],
            x=ds["x"],
            lat=(("y", "x"), ds["geolat"].values),
            lon=(("y", "x"), ds["geolon"].values),
        ),
        dims=("lev", "y", "x"),
    )
    ds[out_v] = xr.DataArray(
        0.5 * ds[v_var][:, :, :-1].values + ds[v_var][:, :, 1:].values,
        coords=dict(
            lev=ds["lev"],
            y=ds["y"],
            x=ds["x"],
            lat=(("y", "x"), ds["geolat"].values),
            lon=(("y", "x"), ds["geolon"].values),
        ),
        dims=("lev", "y", "x"),
    )
    return ds


def combine_tiles(member_tiles, flatten_dim="tile_y_x", coord_dims=("tile", "y", "x")):
    tiles_combined = xr.concat(member_tiles, dim="tile")
    tiles_stacked = tiles_combined.stack(**{flatten_dim: coord_dims})
    return tiles_stacked


def regrid_member(member_tiles, regrid_weights_file):
    tiles_combined = combine_tiles(member_tiles)
    with xr.open_dataset(regrid_weights_file) as regrid_ds:
        # Description of weight file at https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node3.html#SECTION03029300000000000000
        regrid_weights = csr_matrix(
            (regrid_ds["S"].values, (regrid_ds["row"].values, regrid_ds["col"].values)),
            shape=(regrid_ds.sizes["n_a"], regrid_ds.sizes["n_b"]),
        )
        dst_dims = regrid_ds["dst_grid_dims"][::-1]
        lon = regrid_ds["xc_b"].values.reshape(dst_dims)[0]
        lat = regrid_ds["yc_b"].values.reshape(dst_dims)[:, 0]
        lev = tiles_combined["lev"]
        regrid_ds = xr.Dataset(coords=dict(lev=lev, lat=lat, lon=lon))
        ua_var_dim = (regrid_ds["lev"].size, regrid_ds["lat"].size, regrid_ds.lon.size)
        sfc_var_dim = (regrid_ds["lat"].size, regrid_ds["lon"].size)

        for variable in tiles_combined.data_vars:
            if "lev" in member_tiles[0][variable].dims:
                regrid_ds[variable] = xr.DataArray(
                    np.zeros(ua_var_dim, dtype=np.float32),
                    coords=dict(lev=lev, lat=lat, lon=lon),
                    name=variable,
                )
                for lev_index in tiles_combined["lev"]:
                    regrid_ds[variable][lev_index] = (
                        regrid_weights @ tiles_combined[variable][lev_index].values
                    ).reshape(sfc_var_dim)
            else:
                regrid_ds[variable] = xr.DataArray(
                    (regrid_weights @ tiles_combined[variable].values).reshape(
                        sfc_var_dim
                    ),
                    coords=dict(lat=lat, lon=lon),
                    name=variable,
                )
    return regrid_ds


def process_member(
    member,
    member_path=None,
    out_path=None,
    init_date_str=None,
    variables=None,
    weight_file=None,
):
    member_tiles = load_member_tiles(member_path, init_date_str, member, variables)
    if "u_s" in variables and "v_w" in variables:
        for t in range(len(member_tiles)):
            member_tiles[t] = unstagger_winds(member_tiles[t])
    regrid_member(member_tiles, weight_file)

    return
