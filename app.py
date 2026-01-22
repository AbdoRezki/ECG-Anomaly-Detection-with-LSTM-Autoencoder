from flask import Flask, request, jsonify
import torch
from src.model import RecurrentAutoencoder
from src.data_prep import create_sequences  # Adapt for single seq

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecurrentAutoencoder().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
criterion = torch.nn.L1Loss(reduction='sum')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['sequence']  # list of 140 floats
    seq = torch.tensor(data).unsqueeze(0).unsqueeze(1).float().to(device)  # (1,1,140)
    with torch.no_grad():
        recon = model(seq)
        loss = criterion(recon, seq).item()
    threshold = 26  # Tune from val
    is_anomaly = loss > threshold
    return jsonify({'mae_loss': loss, 'is_anomaly': is_anomaly})

if __name__ == '__main__':
    app.run(debug=True)
