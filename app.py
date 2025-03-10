from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from scipy.spatial import Delaunay
from gat import GATClassifier  # Import your GAT model class

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATClassifier(input_dim=32).to(device)  # Adjust input_dim as needed
model.load_state_dict(torch.load("gat_model.pth", map_location=device))
model.eval()

def compute_orb_graph(image):
    orb = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if not keypoints or descriptors is None:
        return None, None, None
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    G = nx.Graph()
    for i, (x, y) in enumerate(points):
        G.add_node(i, pos=(x, y), descriptor=descriptors[i])
    if len(points) > 2:
        tri = Delaunay(points)
        for simplex in tri.simplices:
            for i in range(3):
                G.add_edge(simplex[i], simplex[(i+1) % 3])
    return G, keypoints, descriptors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    G, keypoints, descriptors = compute_orb_graph(image)
    if G is None:
        return jsonify({'error': 'No keypoints detected in image'}), 400
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(np.array([G.nodes[i]['descriptor'] for i in G.nodes]), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index).to(device)
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).item()
    return jsonify({'prediction': 'Real' if pred == 0 else 'Fake'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)


