import osmnx as ox
import time
from shapely.geometry import Polygon
import os
import numpy as np


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # 修复: 直接使用 ox.graph_to_gdfs 而不是 ox.utils_graph.graph_to_gdfs
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # 检查是否存在 _stringify_nonnumeric_cols 函数，如果不存在，提供替代方法
    if hasattr(ox.io, '_stringify_nonnumeric_cols'):
        gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
        gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    else:
        # 替代方法：将非数值列转为字符串
        for gdf in [gdf_nodes, gdf_edges]:
            for col in gdf.columns:
                if gdf[col].dtype == object:
                    gdf[col] = gdf[col].fillna('').astype(str)

    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


print("osmnx version", ox.__version__)

place = "Porto, Portugal"
G = ox.graph_from_place(place, network_type='drive')
start_time = time.time()
save_graph_shapefile_directional(G, filepath='./TTE/data/porto_network')
print("--- %s seconds ---" % (time.time() - start_time))
