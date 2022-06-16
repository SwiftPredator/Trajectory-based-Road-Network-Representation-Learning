from ast import literal_eval

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import swifter
from scipy.ndimage import shift
from shapely import wkt
from shapely.geometry import LineString, box


def preprocess_trajectories_porto(
    df: pd.DataFrame,
    city_bounds: np.array,
    min_gps_points: int = 10,
    polyline_convert: bool = False,
) -> gpd.GeoDataFrame:
    """
    Method to preprocess a trajectorie dataframe containing gps points as string.
    For example the porto trajectorie data. The trajectories are converted to shapely linestrings
    and clipped to the bounding box of the porto street network.
    Args:
        df (pd.DataFrame): dataframe containing the trajectories
        min_gps_points (int, optional): _description_. Defaults to 10.
    """
    if polyline_convert:
        df = convert_polyline(df, min_gps_points)

    df = clip_trajectories(df, city_bounds, polyline_convert=polyline_convert)
    df = filter_min_points(df, min_gps_points)

    return df


def convert_polyline(df: pd.DataFrame, min_gps_points: int) -> pd.DataFrame:
    def convert_to_line_string(x):
        m = np.matrix(x).reshape(-1, 2)
        if m.shape[0] >= min_gps_points:
            line = LineString(m)
            return line
        return -1

    df = df[df["MISSING_DATA"] == False]
    df["POLYLINE"] = df["POLYLINE"].swifter.apply(convert_to_line_string)
    df = df[df["POLYLINE"] != -1]

    return df


def clip_trajectories(
    df: pd.DataFrame, bounds: np.array, polyline_convert: bool = False
) -> gpd.GeoDataFrame:
    bbox = box(*bounds)
    poly_gdf = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326")
    if not polyline_convert:
        df["POLYLINE"] = df["POLYLINE"].swifter.apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs="epsg:4326", geometry="POLYLINE")

    return gdf.clip(poly_gdf, keep_geom_type=True).explode(ignore_index=True)


def filter_min_points(df: gpd.GeoDataFrame, min_gps_points: int) -> gpd.GeoDataFrame:
    df["coords"] = df["POLYLINE"].swifter.apply(lambda x: list(x.coords))
    df = df[df["coords"].str.len() >= min_gps_points]

    return df


"""
Processing Part after gps points are mapped to road segments
"""


def remove_outlier_trajectories(
    df: pd.DataFrame, min_edges_traversed: int = 3, max_speed: float = 1e1
) -> pd.DataFrame:
    df.dropna(inplace=True)
    df["speed"] = df["speed"].swifter.apply(literal_eval)
    df["speed_mean"] = df["speed"].swifter.apply(np.mean)
    df["cpath"] = df["cpath"].swifter.apply(literal_eval)

    # remove mean speed <= 0 since mostly standing trajectories
    # atleast 3 traversed edges & remove remaining zero average speed trajectories
    df = df[(df["cpath"].str.len() >= min_edges_traversed) & (df["speed_mean"] > 0)]

    # smooth trajectories that have inf speed values
    def smooth_inf_and_neg_values(x):
        temp = np.array(x)
        # smoothing
        masks = (temp >= max_speed, temp < 0)
        for mask in masks:
            shift_left, shift_right = shift(mask, -1, cval=0), shift(mask, 1, cval=0)
            temp[mask] = np.sum((shift_right * temp) + shift_left * temp) / np.sum(
                shift_right + shift_left
            )

        return temp

    df[(df["speed_mean"] > max_speed)]["speed"] = df[(df["speed_mean"] > max_speed)][
        "speed"
    ].swifter.apply(smooth_inf_and_neg_values)
    df["speed_mean"] = df["speed"].swifter.apply(np.mean)

    # drop remaining inf trajectories which have more than two consectuive inf values
    df = df[df["speed_mean"] < max_speed]

    assert df[df["speed_mean"] <= 0].shape[0] == 0
    assert df[df["speed_mean"] > max_speed].shape[0] == 0

    return df
