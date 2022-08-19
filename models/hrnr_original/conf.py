beijing_label_hparams = {
    "hidden_dims": 608,
    "node_dims": 512,
    "node_num": 30000,  # 16000 for porto
    "cmt_num": 200,
    "cmt_dims": 256,
    "struct_cmt_num": 300,
    "struct_cmt_dims": 608,
    "fnc_cmt_num": 30,
    "fnc_cmt_dims": 256,
    "label_num": 2,
    "train_cmt_set": "/mnt/data/wuning/RN-GNN/beijing/train_cmt_set",
    "train_loc_set": "/mnt/data/wuning/NTLR/beijing/train_loc_set",
    "train_time_set": "/mnt/data/wuning/NTLR/beijing/train_time_set_eta",
    "adj": "/mnt/data/wuning/NTLR/beijing/CompleteAllGraph",
    "t_adj": "/mnt/data/wuning/RN-GNN/beijing/loc_tra_adj",
    "node_features": "/mnt/data/wuning/RN-GNN/beijing/node_features",
    "label_train_set": "/mnt/data/wuning/RN-GNN/beijing/label_pred_train_set",
    "label_train_set_false": "/mnt/data/wuning/RN-GNN/beijing/label_pred_train_set_false",
    "spectral_label": "/mnt/data/wuning/RN-GNN/beijing/spectral_label",
    "struct_assign": "/mnt/data/wuning/RN-GNN/beijing/spectral_label",
    "fnc_assign": "/mnt/data/wuning/RN-GNN/beijing/fnc_assign_dst",
    "gru_dims": 512,
    "gru_layers": 1,
    "is_bigru": True,
    "state_num": 2,
    "vocab_size": 30000,  # 16000 for porto
    "batch_size": 100,
    "device": 1,
    "use_cn_gnn": False,
    "gnn_layers": 1,
    "eta_epoch": 20,
    "gae_epoch": 1000,
    "eta_learning_rate": 1e-4,
    "gae_learning_rate": 5e-4,
    "g2s_learning_rate": 1e-4,
    "g2t_learning_rate": 1e-4,
    "lp_learning_rate": 1e-4,
    "label_pred_gnn_layer": 2,
    "alpha": 0.2,
    "dropout": 0.6,
    "lane_num": 30,  # 6 for porto
    "length_num": 16836,  # 220 for porto
    "type_num": 20,  # 20 for porto
    "lane_dims": 32,
    "length_dims": 32,
    "type_dims": 32,
    "clip": 0.1,
    "g2s_clip": 1.0,
    "g2t_clip": 1.0,
    "lp_clip": 1.0,
    "g2t_sample_num": 2000,
    "g2t_epoch": 1000,
    "label_epoch": 2000,
    "baseline_gat_layer": 1,
}

beijing_des_hparams = {
    "hidden_dims": 608,
    "node_dims": 512,
    "node_num": 30000,  # 16000 for porto
    "cmt_num": 200,
    "cmt_dims": 256,
    "struct_cmt_num": 300,
    "struct_cmt_dims": 608,
    "fnc_cmt_num": 30,
    "fnc_cmt_dims": 256,
    "train_cmt_set": "data/train_cmt_set",
    "train_loc_set": "data/train_loc_set",
    "train_time_set": "/mnt/data/wuning/NTLR/beijing/train_time_set_eta",
    "adj": "data/CompleteAllGraph",
    "t_adj": "data/loc_tra_adj",
    "node_features": "data/node_features",
    "spectral_label": "data/spectral_label",
    "struct_assign": "/mnt/data/wuning/RN-GNN/beijing/spectral_label",
    "fnc_assign": "/mnt/data/wuning/RN-GNN/beijing/fnc_assign",
    "gru_dims": 512,
    "gru_layers": 1,
    "is_bigru": True,
    "state_num": 2,
    "vocab_size": 30000,  # 16000 for porto
    "batch_size": 100,
    "device": 0,
    "use_cn_gnn": False,
    "gnn_layers": 1,
    "eta_epoch": 20,
    "gae_epoch": 1000,
    "eta_learning_rate": 1e-4,
    "gae_learning_rate": 5e-4,
    "g2s_learning_rate": 1e-4,
    "g2t_learning_rate": 1e-4,
    "lp_learning_rate": 1e-4,
    "loc_pred_gnn_layer": 1,
    "alpha": 0.2,
    "dropout": 0.6,
    "lane_num": 30,  # 6 for porto
    "length_num": 16836,  # 220 for porto
    "type_num": 20,  # 20 for porto
    "lane_dims": 32,
    "length_dims": 32,
    "type_dims": 32,
    "clip": 0.1,
    "g2s_clip": 1.0,
    "g2t_clip": 1.0,
    "lp_clip": 1.0,
    "g2t_sample_num": 2000,
    "g2t_epoch": 1000,
    "baseline_gat_layer": 1,
}


beijing_route_hparams = {
    "hidden_dims": 608,
    "node_dims": 512,
    "node_num": 30000,  # 16000 for porto
    "cmt_num": 200,
    "cmt_dims": 256,
    "struct_cmt_num": 300,
    "struct_cmt_dims": 608,
    "fnc_cmt_num": 30,
    "fnc_cmt_dims": 256,
    "train_cmt_set": "/mnt/data/wuning/RN-GNN/beijing/train_cmt_set",
    "train_loc_set": "/mnt/data/wuning/NTLR/beijing/train_loc_10_set",
    "train_time_set": "/mnt/data/wuning/NTLR/beijing/train_time_set_eta",
    "adj": "/mnt/data/wuning/NTLR/beijing/CompleteAllGraph",
    "t_adj": "/mnt/data/wuning/RN-GNN/beijing/loc_tra_adj",
    "node_features": "/mnt/data/wuning/RN-GNN/beijing/node_features",
    "spectral_label": "/mnt/data/wuning/RN-GNN/beijing/spectral_label",
    "struct_assign": "/mnt/data/wuning/RN-GNN/beijing/spectral_label",
    "fnc_assign": "/mnt/data/wuning/RN-GNN/beijing/fnc_assign",
    "gru_dims": 512,
    "gru_layers": 1,
    "is_bigru": True,
    "state_num": 2,
    "vocab_size": 30000,  # 16000 for porto
    "batch_size": 100,
    "device": 0,
    "use_cn_gnn": False,
    "gnn_layers": 1,
    "eta_epoch": 20,
    "gae_epoch": 1000,
    "eta_learning_rate": 1e-4,
    "gae_learning_rate": 5e-4,
    "g2s_learning_rate": 1e-4,
    "g2t_learning_rate": 1e-4,
    "lp_learning_rate": 1e-4,
    "loc_pred_gnn_layer": 1,
    "alpha": 0.2,
    "dropout": 0.6,
    "lane_num": 30,  # 6 for porto
    "length_num": 16836,  # 220 for porto
    "type_num": 20,  # 20 for porto
    "lane_dims": 32,
    "length_dims": 32,
    "type_dims": 32,
    "clip": 0.1,
    "g2s_clip": 1.0,
    "g2t_clip": 1.0,
    "lp_clip": 1.0,
    "g2t_sample_num": 2000,
    "g2t_epoch": 1000,
    "baseline_gat_layer": 1,
}


beijing_hparams = {
    "hidden_dims": 608,
    "node_dims": 512,
    "node_num": 30000,  # 16000 for porto
    "cmt_num": 200,
    "cmt_dims": 256,
    "struct_cmt_num": 300,
    "struct_cmt_dims": 608,
    "fnc_cmt_num": 30,
    "fnc_cmt_dims": 256,
    "train_cmt_set": "../hrnr_original/data/train_cmt_set",
    "train_loc_set": "../hrnr_original/train_loc_10_set",
    "train_time_set": "/mnt/data/wuning/NTLR/beijing/train_time_set_eta",
    "adj": "../hrnr_original/data/CompleteAllGraph",
    "t_adj": "../hrnr_original/data/loc_tra_adj",
    "node_features": "../hrnr_original/data/node_features",
    "spectral_label": "../hrnr_original/data/spectral_label",
    "struct_assign": "/mnt/data/wuning/RN-GNN/beijing/spectral_label",
    "fnc_assign": "/mnt/data/wuning/RN-GNN/beijing/fnc_assign",
    "gru_dims": 512,
    "gru_layers": 1,
    "is_bigru": True,
    "state_num": 2,
    "vocab_size": 30000,  # 16000 for porto
    "batch_size": 100,
    "device": 1,
    "use_cn_gnn": False,
    "gnn_layers": 1,
    "eta_epoch": 20,
    "gae_epoch": 1000,
    "eta_learning_rate": 1e-4,
    "gae_learning_rate": 5e-4,
    "g2s_learning_rate": 1e-4,
    "g2t_learning_rate": 1e-4,
    "lp_learning_rate": 1e-4,
    "loc_pred_gnn_layer": 1,
    "alpha": 0.2,
    "dropout": 0.6,
    "lane_num": 30,  # 6 for porto
    "length_num": 16836,  # 2200 for porto
    "type_num": 20,  # 20 for porto
    "lane_dims": 32,
    "length_dims": 32,
    "type_dims": 32,
    "clip": 0.1,
    "g2s_clip": 1.0,
    "g2t_clip": 1.0,
    "lp_clip": 1.0,
    "g2t_sample_num": 2000,
    "g2t_epoch": 1000,
    "baseline_gat_layer": 1,
}
