"""
PyTorch ile Sinüs Dalgası Üzerinde RNN (Recurrent Neural Network) Zaman Serisi Tahmini
----------------------------------------------------------------------------------------
Bu modül, PyTorch kullanarak özel bir RNN sınıfı (nn.Module) tanımlar, sinüs dalgası
üzerinde zaman serisi dizilerini eğitir ve gelecek zaman adımı için tahmin gerçekleştirir.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SineRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(SineRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Son zaman adımının çıktısını tam bağlantılı katmana ver
        out = self.fc(out[:, -1, :])
        return out

def generate_sine_data(seq_length=50, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    y = np.sin(x)
    sequences, targets = [], []
    for i in range(len(x) - seq_length):
        sequences.append(y[i:i + seq_length])
        targets.append(y[i + seq_length])
    return np.array(sequences), np.array(targets)

def train_rnn():
    seq_length = 50
    epochs = 15
    batch_size = 32
    learning_rate = 0.001
    
    X, y = generate_sine_data(seq_length=seq_length)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SineRNN(input_size=1, hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("RNN Modeli Eğitiliyor...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Ortalama Loss: {avg_loss:.5f}")
        
    print("Eğitim Tamamlandı.")

if __name__ == "__main__":
    train_rnn()
