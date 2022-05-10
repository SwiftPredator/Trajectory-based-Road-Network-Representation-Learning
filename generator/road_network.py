from dataclasses import dataclass, field

import fmm
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import swifter
from shapely.geometry import LineString


class RoadNetwork:
    """
    Class representing a Road Network.
    """

    G: nx.MultiDiGraph

    def __init__(
        self,
        location: str,
        network_type: str = "roads",
        retain_all: bool = True,
        truncate_by_edge: bool = True,
    ):
        """
        Create network from edge and node file or from a osm query string.

        Args:
            location (str): _description_
            network_type (str, optional): _description_. Defaults to "roads".
            retain_all (bool, optional): _description_. Defaults to True.
            truncate_by_edge (bool, optional): _description_. Defaults to True.
        """
        self.G = ox.graph_from_place(
            location,
            network_type=network_type,
            retain_all=retain_all,
            truncate_by_edge=truncate_by_edge,
        )

    def map_trajectorie(self, coordinates: gpd.GeoDataFrame):
        P = ox.project_graph(self.G)
        B = ox.project_gdf(coordinates)

        X = B.geometry.apply(lambda c: c.centroid.x)
        Y = B.geometry.apply(lambda c: c.centroid.y)

        mapping = ox.nearest_edges(P, X, Y, return_dist=True)

        print(mapping[:10])

    def save(self, path: str):
        """
        Save road network as node and edge shape file.
        Args:
            path (str): file saving path
        """
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.G)
        gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
        gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
        gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype="int")

        gdf_nodes.to_file(path + "/nodes.shp", encoding="utf-8")
        gdf_edges.to_file(path + "/edges.shp", encoding="utf-8")

    def fmm_trajectorie_mapping():

        network = fmm.Network("../osm_data/test.shp")
        graph = fmm.NetworkGraph(network)

        ubodt_gen = fmm.UBODTGenAlgorithm(network, graph)

        ubodt_gen.generate_ubodt("../data/ubodt.txt", 4, binary=False, use_omp=True)
        ubodt = fmm.UBODT.read_ubodt_csv("../data/ubodt.txt")

        model = fmm.FastMapMatch(network, graph, ubodt)

        k = 8
        radius = 0.003
        gps_error = 0.0005
        fmm_config = fmm.FastMapMatchConfig(k, radius, gps_error)

        input_config = fmm.GPSConfig()
        input_config.file = "../datasets/trajectories/Porto/mapped_id_poly.csv"
        input_config.id = "id"
        input_config.geom = "POLYLINE"
        print(input_config.to_string())

        result_config = fmm.ResultConfig()
        result_config.file = "../datasets/trajectories/Porto/mr.txt"
        result_config.output_config.write_opath = True
        result_config.output_config.write_spdist = True
        result_config.output_config.write_speed = True
        print(result_config.to_string())

        status = model.match_gps_file(
            input_config, result_config, fmm_config, use_omp=True
        )
        print(status)

    @staticmethod
    def preprocess_trajectories_porto(
        df: pd.DataFrame, min_gps_points: int = 10
    ) -> pd.DataFrame:
        """
        Static method to preprocess a trajectorie dataframe containing gps points as string.
        For example the porto trajectorie data.
        Args:
            df (pd.DataFrame): dataframe containing the trajectories
            min_gps_points (int, optional): _description_. Defaults to 10.
        """

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

    def visualize():
        ...
        # define the colors to use for different edge types
        # hwy_colors = {'footway': 'skyblue',
        #             'residential': 'paleturquoise',
        #             'cycleway': 'orange',
        #             'service': 'sienna',
        #             'living street': 'lightgreen',
        #             'secondary': 'grey',
        #             'pedestrian': 'lightskyblue'}

        # # return edge IDs that do not match passed list of hwys
        # def find_edges(G, hwys):
        #     edges = []
        #     for u, v, k, data in G.edges(keys=True, data='highway'):
        #         check1 = isinstance(data, str) and data not in hwys
        #         check2 = isinstance(data, list) and all([d not in hwys for d in data])
        #         if check1 or check2:
        #             edges.append((u, v, k))
        #     return set(edges)

        # # first plot all edges that do not appear in hwy_colors's types
        # G_tmp = G.copy()
        # G_tmp.remove_edges_from(G.edges - find_edges(G, hwy_colors.keys()))
        # m = ox.plot_graph_folium(G_tmp, popup_attribute='highway', weight=5, color='black')

        # # then plot each edge type in hwy_colors one at a time
        # for hwy, color in hwy_colors.items():
        #     G_tmp = G.copy()
        #     G_tmp.remove_edges_from(find_edges(G_tmp, [hwy]))
        #     if G_tmp.edges:
        #         m = ox.plot_graph_folium(G_tmp,
        #                                 graph_map=m,
        #                                 popup_attribute='highway',
        #                                 weight=5,
        #                                 color=color)
        # m
