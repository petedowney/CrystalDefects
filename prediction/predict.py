
import graph_creation
import torch
import GCN

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

LABELS = [
    ('hse-ex1', 'defect_hse_excited_deltaQ', '4.0'),
    ("hse-ex1", "defect_zero_phonon_line", '4.0'), 
    ('hse-gs', 'defect_lvl_tdm', '4.0'),
    ('hse-gs', 'defect_total_energy', '4.0'), 
    ('hse-gs', 'defect_fermi_energy', '4.0'),
    ('pbe-gs', 'defect_cbm-vbm_energy', '4.0'),
    ('pbe-gs', 'defect_initial_pbe_deltaQ', '4.0'),
    ('pbe-gs', 'defect_initial_pbe_deltaR', '4.0'),
    ('pbe-gs', 'defect_electron_affinity', '4.0'),
    ('pbe-gs', 'defect_ionization_energy', '4.0')
]

NORMS = {
    "defect_hse_excited_deltaQ": (0.000721, 0.020286),
    "defect_zero_phonon_line": (0.300373, 1.173601),
    "defect_lvl_tdm": (0.007010, 18.534500),
    "defect_total_energy": (-1441.350724, -1410.170806),
    "defect_fermi_energy": (5.529372, 6.463675),
    "defect_cbm-vbm_energy": (0.038700, 0.361800),
    "defect_initial_pbe_deltaQ": (0.033092, 1.463129),
    "defect_initial_pbe_deltaR": (0.001019, 0.033140),
    "defect_electron_affinity": (5.884099, 6.473329),
    "defect_ionization_energy": (5.884099, 6.473329),
}


def predict(poscar_file, charge):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_matrix, feature_vector = graph_creation.get_graph_data(poscar_file, charge)

    predicted_properties = {}
    for label1, label2, version in LABELS:
        path = f'models/model_{label1}_{label2}{version}.pt'
        predicted_properties[label2] = predict_property(adj_matrix, feature_vector, device, path) * (NORMS[label2][1] - NORMS[label2][0]) + NORMS[label2][0]

    return predicted_properties

def predict_property(adj_matrix, feature_vector, device, model_path):
    model_instance = GCN.GraphConv(20, 0.1)
    model_instance.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model_instance.to(device)
    model_instance.eval()

    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

    edge_index, edge_weight = dense_to_sparse(adj_matrix)

    feature_vector = torch.tensor(feature_vector, dtype=torch.float32).to(device)

    data = Data(x=feature_vector, edge_index=edge_index, edge_weight=edge_weight)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        output = model_instance(data.x, data.edge_index, data.edge_weight, batch)

    return output.cpu().numpy().squeeze()



        # # Prepare validation dataset for prediction
        # val_graphs = []
        # for i in range(len(X_val_tensor)):
        #     edge_index, _ = dense_to_sparse(adj_val_tensor[i, :GCN.GRAPH_SIZE, :GCN.GRAPH_SIZE])
        #     data = Data(x=X_val_tensor[i, :GCN.GRAPH_SIZE], edge_index=edge_index)
        #     val_graphs.append(data)

        # # Predict on validation set
        # preds = []
        # with torch.no_grad():
        #     for data in val_graphs:
        #         x = data.x
        #         edge_index = data.edge_index
        #         batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        #         pred = model(x, data.edge_index, data.edge_weight, batch)
        #         preds.append(pred.cpu().numpy().flatten()[0])

