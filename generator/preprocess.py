import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import swifter
from shapely import wkt
from shapely.geometry import LineString, box


def preprocess_trajectories_porto(
    self,
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

    df = clip_trajectories(df, city_bounds)
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


def clip_trajectories(df: pd.DataFrame, bounds: np.array) -> gpd.GeoDataFrame:
    bbox = box(*bounds)
    poly_gdf = gpd.GeoDataFrame([1], geometry=[bbox], crs="EPSG:4326")

    df["POLYLINE"] = df["POLYLINE"].swifter.apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs="epsg:4326", geometry="POLYLINE")

    return gdf.clip(poly_gdf, keep_geom_type=True).explode(ignore_index=True)


def filter_min_points(df: gpd.GeoDataFrame, min_gps_points: int) -> gpd.GeoDataFrame:
    df["coords"] = df["POLYLINE"].swifter.apply(lambda x: list(x.coords))
    df = df[len(df["coords"]) >= min_gps_points]

    return df
