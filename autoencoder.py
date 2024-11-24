import torch
import torch.nn as nn
import torch.optim as optim
from utils import read_episode_file

class Autoencoder(nn.Module):
    def __init__(self, input_size=658, embedding_size=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def hex_to_binary_vector(hex_str, output_length=400):
    binary_str = bin(int(hex_str, 16))[2:]
    binary_str = binary_str.zfill(output_length)
    binary_vector = torch.tensor([int(bit) for bit in binary_str], dtype=torch.float32)
    return binary_vector

input_size = 658
embedding_size = 32
batch_size = 2048
epochs = 5000
learning_rate = 0.003

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
data = read_episode_file("./episodes/test/episodes.txt")[0]
binary_data = [item[1] for item in data] + [item[2] for item in data]
print(len(binary_data))
binary_data = torch.tensor(
    [[int(bit) for bit in binary_str] for binary_str in binary_data], dtype=torch.float32
).to(device)
train_loader = torch.utils.data.DataLoader(binary_data, batch_size=batch_size, shuffle=True)

model = Autoencoder(input_size=input_size, embedding_size=embedding_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler()

for epoch in range(epochs):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        embeddings, outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    if loss.item() < 0.01:
        break

print("Training completed!")

def binary_to_hex(binary_vector, output_length=input_size):
    binary_list = binary_vector.flatten().tolist()
    binary_string = ''.join(str(int(bit)) for bit in binary_list)
    binary_string = binary_string.zfill(output_length)
    hex_string = hex(int(binary_string, 2))[2:]
    hex_length = output_length // 4
    hex_string = hex_string.zfill(hex_length)
    return hex_string

sample_hex = "000200080000000080000100000000807f800000000210843834bfaf4cb388ad0a010400000002b5ad6a0056b5ad400108421005def7bd003bdef7a0000150008042008001000740020000e80040084210802"
sample_binary = hex_to_binary_vector(sample_hex, output_length=input_size).unsqueeze(0).to(device)
print(sample_binary.shape)
embedding, reconstruction = model(sample_binary)
reconstructed_binary = torch.round(reconstruction).squeeze(0)
reconstructed_hex = binary_to_hex(reconstructed_binary)

print(f"Input Hex: {sample_hex}")
print(f"Embedding: {embedding}")
print(f"Reconstructed Binary as Hex: {reconstructed_hex}")