def generate_trajid_to_nodeid(network):
    map = {}
    nodes = list(network.line_graph.nodes)
    for index, id in zip(network.gdf_edges.index, network.gdf_edges.fid):
        map[id] = nodes.index(index)

    return map
