from dataclasses import dataclass, field
from typing import List

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import swifter
from shapely.geometry import LineString
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.data import Data
except ImportError:
    ...

try:
    import fmm
    from fmm import (STMATCH, FastMapMatchConfig, Network, NetworkGraph,
                     STMATCHConfig)
except ImportError:
    ...


class RoadNetwork:
    """
    Class representing a Road Network.
    """

    G: nx.MultiDiGraph
    gdf_nodes: gpd.GeoDataFrame
    gdf_edges: gpd.GeoDataFrame

    def __init__(
        self,
        location: str = None,
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
        if location != None:
            self.G = ox.graph_from_place(
                location,
                network_type=network_type,
                retain_all=retain_all,
                truncate_by_edge=truncate_by_edge,
            )
            self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.G)

    def save(self, path: str):
        """
        Save road network as node and edge shape file.
        Args:
            path (str): file saving path
        """
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.G)
        gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
        gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
        gdf_edges["fid"] = np.arange(
            0, gdf_edges.shape[0], dtype="int"
        )  # id for each edge

        gdf_nodes.to_file(path + "/nodes.shp", encoding="utf-8")
        gdf_edges.to_file(path + "/edges.shp", encoding="utf-8")

    def load(self, path):
        """
        Load graph from edges and nodes shape file
        """
        self.gdf_nodes = gpd.read_file(path + "/nodes.shp")
        self.gdf_edges = gpd.read_file(path + "/edges.shp")
        self.gdf_nodes.set_index("osmid", inplace=True)
        self.gdf_edges.set_index(["u", "v", "key"], inplace=True)

        # encode highway column
        self.gdf_edges["highway"] = self.gdf_edges["highway"].str.extract(r"(\w+)")
        le = LabelEncoder()
        self.gdf_edges["highway_enc"] = le.fit_transform(self.gdf_edges["highway"])

        self.G = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)

    def fmm_trajectorie_mapping(
        self, network_file: str, input_file: str, output_files: str
    ):
        """
        Maps raw trajectory gps data to corresponding road segments on the osmnx graph
        """

        network = Network(network_file, "fid", "u", "v")
        graph = NetworkGraph(network)

        stmatch_model = STMATCH(network, graph)

        # ubodt_gen = fmm.UBODTGenAlgorithm(network, graph)

        # ubodt_gen.generate_ubodt("ubodt.txt", 0.03, binary=False, use_omp=True)
        # ubodt = fmm.UBODT.read_ubodt_csv("ubodt.txt")

        # fmm_model = fmm.FastMapMatch(network, graph, ubodt)

        k = 16
        gps_error = 0.0005
        radius = 0.003
        vmax = 0.0003
        factor = 1.5
        stmatch_config = STMATCHConfig(k, radius, gps_error, vmax, factor)
        # fmm_config = FastMapMatchConfig(k, radius, gps_error)

        input_config = fmm.GPSConfig()
        input_config.file = input_file
        input_config.id = "id"
        input_config.geom = "POLYLINE"
        input_config.timestamp = "timestamp"
        print(input_config.to_string())

        result_config = fmm.ResultConfig()
        result_config.file = output_files
        result_config.output_config.write_opath = True
        result_config.output_config.write_ogeom = True
        result_config.output_config.write_pgeom = True
        result_config.output_config.write_spdist = True
        result_config.output_config.write_speed = True
        result_config.output_config.write_duration = True
        print(result_config.to_string())

        status = stmatch_model.match_gps_file(
            input_config, result_config, stmatch_config, use_omp=True
        )
        print(status)

    @property
    def bounds(self):
        return self.gdf_nodes.geometry.total_bounds

    @property
    def line_graph(self):
        return nx.line_graph(self.G, create_using=nx.DiGraph)

    def generate_road_segment_pyg_dataset(
        self,
        traj_data: gpd.GeoDataFrame = None,
        drop_labels: List = [],
        include_coords: bool = False,
    ):
        """
        Generates road segment feature dataset in the pyg Data format.
        if traj_data given it will also generate trajectory based features
        like avg. speed and avg. utilization on each road segment
        """
        # create edge_index for line
        LG = self.line_graph
        # create edge_index
        map_id = {j: i for i, j in enumerate(LG.nodes)}
        edge_list = nx.to_pandas_edgelist(LG)
        edge_list["sidx"] = edge_list["source"].map(map_id)
        edge_list["tidx"] = edge_list["target"].map(map_id)

        edge_index = np.array(edge_list[["sidx", "tidx"]].values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        # create feature matrix
        df = self.gdf_edges.copy()
        df["idx"] = df.index.map(map_id)
        df.sort_values(by="idx", axis=0, inplace=True)

        df.rename(columns={"fid": "id"}, inplace=True)
        if traj_data is not None:
            # incorperate trajectorie data in form of speed and volume
            traj_data.drop(["id"], axis=1, inplace=True)
            df = df.join(traj_data)

        # print(df.head()

        if include_coords:
            df["x"] = df.geometry.centroid.x / 100  # normalize to -2/2
            df["y"] = df.geometry.centroid.y / 100  # normalize to -1/1

        df.drop(
            [
                "osmid",
                "id",
                "geometry",
                "highway",
                "idx",
                "name",
                "ref",
                "access",
                "area",
                "width",
            ],
            axis=1,
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)
        df["bridge"] = (
            df["bridge"]
            .fillna(0)
            .replace(["yes", "viaduct", "['yes', 'viaduct']", "cantilever"], 1)
        )
        df["tunnel"] = (
            df["tunnel"].fillna(0).replace(["yes", "building_passage", "culvert"], 1)
        )
        df["junction"] = df["junction"].fillna(0).replace(["roundabout", "circular"], 1)
        df["lanes"] = df["lanes"].str.extract(r"(\w+)")
        df["maxspeed"] = df["maxspeed"].str.extract(r"(\w+)")

        imputer = KNNImputer(n_neighbors=1)
        imputed = imputer.fit_transform(df)
        df["lanes"] = imputed[:, 2].astype(int)
        df["maxspeed"] = imputed[:, 3].astype(int)

        df.drop(drop_labels, axis=1, inplace=True)  # drop label?

        # normalize continiuos features
        df["length"] = (df["length"] - df["length"].min()) / (
            df["length"].max() - df["length"].min()
        )  # min max normalization

        # Categorical features one hot encoding
        df = pd.get_dummies(
            df,
            columns=list(
                set(["highway_enc", "lanes", "maxspeed"]).symmetric_difference(
                    set(drop_labels)
                )
            ),
            drop_first=True,
        )
        # print(df.head(), df.shape)

        features = torch.DoubleTensor(np.array(df.values, dtype=np.double))
        # print(features)
        # create pyg dataset
        data = Data(x=features, edge_index=edge_index)
        transform = T.Compose(
            [
                T.NormalizeFeatures(),
            ]
        )
        data = transform(data)

        return data

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
