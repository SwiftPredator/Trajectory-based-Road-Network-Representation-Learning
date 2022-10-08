from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class TemporalEmbeddingPlugin:
    def __init__(self, model: nn.Module, network, device):
        super().__init__()
        self.model = model
        self.device = device
        self.network = network
        self.model.eval()

    def generate_emb(self, times):
        # Bx3 - 3 time features B -> Batch Size
        timeframe_batch = torch.zeros(
            times.shape[0],
            12,
            self.processed_temp_data.shape[1],
            self.processed_temp_data.shape[2],
        )
        for i, t in enumerate(times):
            start_idx, end_idx = TemporalEmbeddingPlugin.time_to_index(t)
            timeframe = self.processed_temp_data[start_idx:end_idx, :, :]
            # pad with zeros if timeframe would normally go beyong
            # if timeframe.shape[0] < 12:
            #     timeframe[timeframe.shape[0]+1:12, :, :] = 0
            timeframe_batch[i, :, :, :] = timeframe

        with torch.no_grad():
            return self.model(timeframe_batch.to(self.device))

    def generate_data(self, temporal_data):
        self.processed_temp_data = TemporalEmbeddingPlugin.prepare_temp_data(
            temporal_data, self.network
        )
        self.processed_temp_data = torch.swapaxes(self.processed_temp_data, 0, 1)
        # norm data
        for i in range(self.processed_temp_data.shape[-1]):
            max_val = self.processed_temp_data[:, :, i].max()
            self.processed_temp_data[:, :, i] = (
                self.processed_temp_data[:, :, i] / max_val
            )

    def load_data(self, path):
        self.processed_temp_data = torch.load(path)
        self.processed_temp_data = torch.cat(
            [self.processed_temp_data, self.processed_temp_data[: 24 * 4]], dim=0
        )
        # norm data
        for i in range(self.processed_temp_data.shape[-1]):
            max_val = self.processed_temp_data[:, :, i].max()
            self.processed_temp_data[:, :, i] = (
                self.processed_temp_data[:, :, i] / max_val
            )

    @staticmethod
    def prepare_temp_data(temporal, network):
        # General temporal data (Note: Nodes are ordered by line graph)
        temporal["time"] = pd.to_datetime(temporal["time"])
        max_steps, min_steps = temporal["time"].max(), temporal["time"].min()
        pad = 0
        data = []
        for i, index in tqdm(enumerate(network.line_graph.nodes)):
            row = network.gdf_edges.loc[index]
            temp = temporal[temporal["id"] == row["id"]][["time", "speed"]].sort_values(
                "time"
            )
            temp = temp.set_index("time")
            if min_steps not in temp.index:
                temp.loc[min_steps] = pad
            if max_steps not in temp.index:
                temp.loc[max_steps] = pad
            temp = temp.asfreq("15Min", fill_value=0)
            temp_avg = temp.groupby(
                by=[temp.index.dayofweek, temp.index.hour, temp.index.minute]
            ).mean()
            temp_avg[["length", "speed_limit", "highway_enc"]] = row[
                ["length", "speed_limit", "highway_enc"]
            ]
            data.append(temp_avg.values)

        return torch.Tensor(data)

    @staticmethod
    def time_to_index(time):
        hours_per_day = 24
        interval_per_hour = 4
        start_idx = (
            time[0] * hours_per_day * interval_per_hour
        ) + interval_per_hour * time[1]
        end_idx = (
            (time[0] * hours_per_day * interval_per_hour)
            + interval_per_hour * time[1]
            + 12
        )  # three hours i.e 12 steps

        return start_idx, end_idx


class NormalPlugin:
    def __init__(self, model: nn.Module, network, device):
        super().__init__()
        self.model = model
        self.device = device
        self.network = network

    def generate_emb(self, times):
        # Bx3 - 3 time features B -> Batch Size
        return torch.Tensor(self.model.load_emb()).unsqueeze(0), None
