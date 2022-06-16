import collections
from math import *

import networkx as nx
import numpy as np
import osmnx as ox
from mxnet import gpu, nd
from mxnet.gluon.nn import Activation, Dense, HybridBlock, HybridSequential


class Identity(HybridBlock):
    def hybrid_forward(self, F, X):
        return X


def get_activation(activation):
    if activation is None:
        return Identity()
    elif type(activation) is str:
        return Activation(activation)
    else:
        return activation


class FeedForward(HybridSequential):
    def __init__(self, units, in_units, activation=Identity(), *args, **kwargs):
        super().__init__(**kwargs)

        dense = Dense(units=units, in_units=in_units, activation=None, *args, **kwargs)
        super().add(dense)
        super().add(activation)

        self.out_units = self.units = units


class NoFeedForward(Identity):
    pass


"""
Following part adds functions to preprocess the street network as input for rfn
"""


def latlng2dist(v1, v2):
    R = 6373.0
    lat1, lon1 = v1
    lat2, lon2 = v2
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def add_edge_position_info(G: nx.MultiDiGraph):
    for edge in list(G.edges(keys=True, data=True)):
        u, v, k, info = edge
        u_pos_x = G.nodes[u]["x"]
        u_pos_y = G.nodes[u]["y"]
        v_pos_x = G.nodes[v]["x"]
        v_pos_y = G.nodes[v]["y"]
        info["u_coord"] = (u_pos_x, u_pos_y)
        info["v_coord"] = (v_pos_x, v_pos_y)
        if "geometry" in info:
            geom_coords = list(info["geometry"].coords)
            info["u_next_coord"] = geom_coords[1]
            info["v_prev_coord"] = geom_coords[-2]
        else:
            info["u_next_coord"] = (v_pos_x, v_pos_y)
            info["v_prev_coord"] = (u_pos_x, u_pos_y)


def make_dual_graph(G: nx.MultiDiGraph):
    D = nx.DiGraph()
    D.add_nodes_from(edge for edge in G.edges)

    for e in G.edges(keys=True):
        u, v, k = e
        for o in G.out_edges(v, keys=True):
            if e != o:
                D.add_edge(e, o)

    # copy edge attrib
    for e in D.nodes(data=True):
        edge, info = e
        u, v, k = edge
        for a, b in G.edges[u, v, k].items():
            info[a] = b

    return D


def add_between_edge_attrib_custom(D, N=4):
    X_B = np.zeros(shape=(D.number_of_edges(), N))
    theta_histogram = [0 for _ in range(N)]
    index = 0
    for between_edge in D.edges(data=True):
        edge1, edge2, info = between_edge
        # print(G.edges[edge1]['v_coord'])
        assert D.nodes[edge1]["v_coord"][0] == D.nodes[edge2]["u_coord"][0]
        assert D.nodes[edge1]["v_coord"][1] == D.nodes[edge2]["u_coord"][1]

        p1 = D.nodes[edge1]["v_prev_coord"]
        p2 = D.nodes[edge1]["v_coord"]
        p3 = D.nodes[edge2]["u_next_coord"]

        c = latlng2dist(p1, p2)
        a = latlng2dist(p2, p3)
        b = latlng2dist(p3, p1)

        AB = (p2[0] - p1[0], p2[1] - p1[1])
        BC = (p3[0] - p2[0], p3[1] - p2[1])
        cr = AB[0] * BC[1] - AB[1] * BC[0]

        if b != 0 and a != 0 and c != 0:
            cosB = (a * a + c * c - b * b) / (2 * a * c)
            cosB = min(max(cosB, -1), 1)
            theta = degrees(acos(cosB))
            theta = 180.0 - theta
            theta *= -1 if cr < 0 else 1
        else:
            theta = 180

        U = 360.0 / N

        theta = (theta + 0.5 * U + 360) % 360
        theta_i = int((theta / 360.0) * N)
        info["turning_angle"] = theta_i
        X_B[index, theta_i] = 1.0
        theta_histogram[theta_i] += 1
        index += 1

    print(theta_histogram)
    return X_B


def add_between_edge_attrib(D, N=4):
    X_B = np.zeros(shape=(D.number_of_edges(), N))
    theta_histogram = [0 for _ in range(N)]
    index = 0
    for between_edge in D.edges(data=True):
        edge1, edge2, info = between_edge
        b1 = D.nodes[edge1]["bearing"]
        b2 = D.nodes[edge2]["bearing"]
        theta = b2 - b1
        if np.isnan(b1):
            theta = 0
        elif np.isnan(b2):
            theta = 0

        U = 360.0 / N
        theta = (theta + 0.5 * U + 360) % 360
        theta_i = int((theta / 360.0) * N)
        info["turning_angle"] = theta_i
        X_B[index, theta_i] = 1.0
        theta_histogram[theta_i] += 1
        index += 1

    print(theta_histogram)
    return X_B


def get_edge_attrib(G: nx.MultiDiGraph):
    check_lists = [
        "residential",
        "unclassified",
        "tertiary",
        "tertiary_link",
        "trunk_link",
        "primary",
        "secondary",
        "living_street",
        "primary_link",
        "secondary_link",
        "motorway_link",
        "trunk",
        "motorway",
        "road",
        "bus_guideway",
    ]

    X_E = np.zeros(shape=(G.number_of_edges(), len(check_lists) + 1), dtype=np.float32)
    for edge_index, edge in enumerate(G.edges(keys=True, data=True)):
        u, v, k, info = edge
        highway_type = info["highway"]
        i = 0
        for valid_highway_type in check_lists:
            if type(highway_type) is list:
                if valid_highway_type in highway_type:
                    X_E[edge_index, i] = 1
            else:
                if valid_highway_type == highway_type:
                    X_E[edge_index, i] = 1
            i += 1
        X_E[edge_index, i] = info["length"] / 100.0

    return X_E


def get_vertex_attrib(G: nx.MultiDiGraph):
    X_V = np.zeros(shape=(G.number_of_nodes(), 2), dtype=np.float32)
    index = 0
    for node in G.nodes(data=True):
        u, info = node
        X_V[index][0] = G.in_degree(u)
        X_V[index][1] = G.out_degree(u)
        index += 1
    return X_V / 6


def get_edge_target_y(G: nx.MultiDiGraph, target_value="speed_kph"):
    _, edges = ox.graph_to_gdfs(G, nodes=True)
    return edges[target_value].values / 60


def load_city_graph(city_name):
    G = ox.load_graphml("data/%s_drive_network_original.graphml" % city_name)
    G = nx.convert_node_labels_to_integers(G, ordering="default")
    G = ox.add_edge_bearings(G)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    # for e in G.edges(data=True):
    #     u, v, info = e
    #     if type(info['highway']) == list:
    #         info['highway'] = ":".join(info['highway'])
    return G


def generate_required_city_graph(city_name, G):
    add_edge_position_info(G)
    D = make_dual_graph(G)

    X_B_np = add_between_edge_attrib(D)
    X_E_np = get_edge_attrib(G)
    X_V_np = get_vertex_attrib(G)
    y_np = get_edge_target_y(G, target_value="speed_kph")

    print(
        "Primal V,E: (%d, %d), Dual V,E: (%d, %d)"
        % (
            G.number_of_nodes(),
            G.number_of_edges(),
            D.number_of_nodes(),
            D.number_of_edges(),
        )
    )

    for _, n in enumerate(D.nodes(data=True)):
        u, info = n
        info["edge_info"] = u

    primal_graph = G
    dual_graph = D

    X_V = nd.array(X_V_np, ctx=gpu())
    X_E = nd.array(X_E_np, ctx=gpu())
    X_B = nd.array(X_B_np, ctx=gpu())
    y = nd.array(y_np, ctx=gpu())

    rfncity = RFNCity(city_name, primal_graph, dual_graph)
    rfncity.set_features(X_V, X_E, X_B, y)

    return rfncity


class City:
    pass


class RFNCity(City):
    def __init__(self, name, primal_graph, dual_graph):
        self.name = name

        node_indices_primal = {
            node: idx for idx, node in enumerate(primal_graph.nodes())
        }
        edge_indices_primal = {
            edge: idx for idx, edge in enumerate(primal_graph.edges())
        }

        (
            N_node_primal,
            N_edge_primal,
            N_mask_primal,
        ) = edge_neighborhoods_primal = make_neighborhood_matrices(
            primal_graph, node_indices_primal, edge_indices_primal
        )

        self.N_node_primal = N_node_primal
        self.N_edge_primal = N_edge_primal
        self.N_mask_primal = N_mask_primal

        node_indices_dual = {node: idx for idx, node in enumerate(dual_graph.nodes())}
        edge_indices_dual = {edge: idx for idx, edge in enumerate(dual_graph.edges())}

        (N_node_dual, N_edge_dual, N_mask_dual), (
            N_common_node,
            N_common_node_mask,
        ) = make_neighborhood_matrices(
            dual_graph,
            node_indices_dual,
            edge_indices_dual,
            is_dual=True,
            node_indices_primal=node_indices_primal,
        )

        self.N_node_dual = N_node_dual
        self.N_edge_dual = N_edge_dual
        self.N_mask_dual = N_mask_dual
        self.N_common_node = N_common_node
        self.N_common_node_mask = N_common_node_mask

    def set_features(self, X_V, X_E, X_B, y):
        self.X_V = X_V
        self.X_E = X_E
        self.X_B = X_B
        self.y = y


def make_neighborhood_matrices(
    graph, node_indices, edge_indices, is_dual=False, node_indices_primal=None
):
    """
    Converts a graph into node and edge the required neighborhood matrices with asso
    Args:
        graph: A networkx primal or dual graph presentation of a road network.
        node_indices: A map from node objects in the graph to node indexes in
                      the output node neighborhood matrix N_node.
        edge_indices: A map from edge objects in the graph to edge indexes in
                      the output edge neighborhood matrix N_edge.
        is_dual: Boolean flag that indicates whether the graph is a dual graph.
        node_indices_primal: Must be supplied if is_dual is True.
                             Maps node objects in the primal graph to node indexes in
                             the common node neighborhood matrix N_common_node.
    Returns:
        N_node: A node adjacency list in matrix format.
                The ith row contains the node indices of the nodes
                in the neighborhood of the ith node. of the ith node.
        N_edge: A node adjacency list in matrix format.
                The ith row contains the edge indices of the edges connecting the ith
                node to its neighbors.
        N_mask: A matrix that indicates whether the jth entry in N_node or N_edge
                exists in the graph.
        N_common_node: Only returned if is_dual=True.
                       A common node is a node that is common to the two edges
                       connected by a between-edge. The ith row and jth column in
                       this matrix contains the index of the common node that connects
                       ith edge (a node in the dual graph) is connected to its jth neighbor.
        N_common_mask: Only returned if is_dual=True.
                       Similar to N_mask, N_common_mask indicates whether the jth entry in
                       N_common_node exists in the graph.
    Raises:
        KeyError: Raises an exception.
    """
    assert not is_dual or is_dual and node_indices_primal is not None
    N_node = []
    N_edge = []

    nodes = sorted(list(graph.nodes()), key=lambda node: node_indices[node])

    for node in nodes:
        node_neighbors = []
        edge_neighbors = []

        predecessors = graph.predecessors(node)
        for neighbor in predecessors:
            node_neighbors.append(node_indices[neighbor])
            edge = (neighbor, node)
            edge_neighbors.append(edge_indices[edge])

        successors = graph.successors(node)
        for neighbor in successors:
            node_neighbors.append(node_indices[neighbor])
            edge = (node, neighbor)
            edge_neighbors.append(edge_indices[edge])

        assert len(node_neighbors) == len(edge_neighbors)
        N_node.append(node_neighbors)
        N_edge.append(edge_neighbors)

    N_node, N_mask = mask_neighborhoods(N_node)
    N_edge, _ = mask_neighborhoods(N_edge)
    N_mask = N_mask.reshape(*N_node.shape[:2], 1)

    if is_dual:
        N_common_node = [[node_indices_primal[edge[0][1]]] for edge in graph.edges()]
        N_common_node, N_common_node_mask = mask_neighborhoods(N_common_node, is_dual)
        N_common_node_mask = N_common_node_mask.reshape(*N_common_node.shape[:2], 1)
        return (N_node, N_edge, N_mask), (N_common_node, N_common_node_mask)
    else:
        return N_node, N_edge, N_mask


def mask_neighborhoods(neighborhoods_list, is_dual=False):
    max_no_neighbors = max(len(n) for n in neighborhoods_list) if not is_dual else 1
    shape = (len(neighborhoods_list), max_no_neighbors)

    neighborhoods_array = nd.zeros(shape=shape, dtype=np.int32, ctx=gpu())
    mask = nd.zeros(shape=shape, ctx=gpu())

    for idx, neighborhood in enumerate(neighborhoods_list):
        neighborhood_size = len(neighborhood)
        if neighborhood_size == 0:
            mask[:] = 1
            continue
        else:
            neighborhoods_array[idx][:neighborhood_size] = neighborhood
            mask[idx][:neighborhood_size] = 1

    return neighborhoods_array, mask
