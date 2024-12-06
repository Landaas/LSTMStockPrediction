import torch
from torch.utils.data import Dataset, DataLoader
from load import X, y

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_size = int(len(X_tensor) * 0.7)
val_size = int(len(X_tensor) * 0.15)
test_size = len(X_tensor) - train_size - val_size

X_train, X_val, X_test = torch.split(X_tensor, [train_size, val_size, test_size])
y_train, y_val, y_test = torch.split(y_tensor, [train_size, val_size, test_size])

train_dataset = StockDataset(X_train, y_train)
val_dataset = StockDataset(X_val, y_val)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
