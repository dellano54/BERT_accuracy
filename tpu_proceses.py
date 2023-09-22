import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Define a custom dataset (you can replace this with your own dataset)
class CustomDataset(Dataset):
    def __init__(self, size, length):
        self.data = torch.randn(size, length)
        self.target = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# Define the training function
def train(rank, dataset):
    # Initialize the TPU device
    device = xm.xla_device()

    # Create a model and move it to the TPU device
    model = SimpleModel().to(device)

    # Split the dataset across all cores using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):
        for batch_data, batch_target in train_loader:
            optimizer.zero_grad()
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            xm.optimizer_step(optimizer)

    # Save the model (only on the first core)
    if rank == 0:
        xser.save(model.state_dict(), 'model.pth')

# Main function for distributed training
def main():
    # Initialize XLA
    xmp.spawn(train, args=(CustomDataset(1000, 10),))

if __name__ == "__main__":
    main()
