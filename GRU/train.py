from GRUModel import GRUModel
from load import X
import torch
from dataset import train_loader, val_loader

input_size = X.shape[2]
output_size = 1
learning_rate = 0.001
epochs = 10 

hidden_sizes = [64, 128, 256]
num_layers_list = [1, 2]

criterion = torch.nn.L1Loss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(f"Training model with hidden_size={hidden_size}, num_layers={num_layers} for {epochs} epochs")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save(model, f'models/model_hs={hidden_size}_layers={num_layers}_epochs={epochs}_batch=16.pth')
        print(f"Saved model_hs={hidden_size}_layers={num_layers}_epochs={epochs}_batch=16.pth\n")
