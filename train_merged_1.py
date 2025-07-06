import json
import os
import random
import data_tool1
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from modules import Xfeatmodel1, CPCmodel_PPOGC_NC, hum, CPCmodel_PPOGC, NAXfeatmodel

import math




class SequentialChunkedDataset(IterableDataset):
    def __init__(self, folder_path, shuffle_files=False):
        self.folder_path = folder_path
        self.shuffle_files = shuffle_files
        self.json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if self.shuffle_files:
            random.shuffle(self.json_files)  # The order of files can be shuffled, but the temporal sequence within each file remains fixed

    def process_file(self, json_file):
        """Process a single file and return the generated data (keep time series fixed)."""
        file_path = os.path.join(self.folder_path, json_file)
        data = data_tool1.load_and_fix_json(file_path)
        if data is None:
            return []
        # processed_data = data_tool.process_data(data)
        converted_data = convert_state_to_tensor(data)
        
        # Call hum function to generate features
        hum_features = hum.count_occurrences(data)

        # Debug: Check tensor shapes
        # for state_item, hum_feature in zip(converted_data, hum_features):
        #     print("state_tensor shape:", state_item["state_tensor"].shape)
            # print("hum_feature shape:", hum_feature.shape)

        # Combine state_tensor and hum_feature
        combined_data = []
        for state_item, hum_feature in zip(converted_data, hum_features):
            # print("111", type(hum_feature["p2"]))
            # print("222", hum_feature["feat"])
            state_item["hum_feature"] = hum_feature["feat"]
            combined_data.append(state_item)

        return combined_data


    def __iter__(self):
            """Iterator: load data file by file"""
            for json_file in self.json_files:
                file_data = self.process_file(json_file)
                for item in file_data:
                    yield item  # Return a dictionary containing state_tensor and hum_feature

def convert_state_to_tensor(data):
    """Convert data to tensor format."""
    converted_data = []
    for step_info in data:
        state = step_info["state"]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.permute(1, 0).view(12, 16, 16)
        converted_data.append({"step": step_info["step"], "state_tensor": state_tensor})
    return converted_data


def custom_collate(batch):
    """Custom collate function to handle batch data."""
    state_tensors = [item["state_tensor"] for item in batch]
    hum_features = [item["hum_feature"] for item in batch]
    return {
        "state_tensor": torch.stack(state_tensors),
        "hum_feature": torch.stack(hum_features)
    }

def train_sequential_model(ckpt_save_path, batch_size=100, n_epochs=10, lr=3e-4, gamma_steplr=0.5, device_num='0', checkpoint_path=None):
    """Train sequential model with CPC loss."""
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    # Initialize new model
    encoder = Xfeatmodel1.XFeatModel().to(device)
    cdc_model = CPCmodel_PPOGC.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=encoder).to(device)
    
    # Optimizer and Learning Rate Scheduler
    optimizer = Adam(list(encoder.parameters()) , lr=lr)
    scheduler = StepLR(optimizer, step_size=n_epochs // 10, gamma=gamma_steplr)

    # If checkpoint path is provided, load model state
    start_epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        cdc_model.load_state_dict(checkpoint['cdc_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch}.")

    # Create dataset
    dataset = SequentialChunkedDataset(folder_path="/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/JL/PPOGC", shuffle_files=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,  # Use custom collate function
        drop_last = True
    )

    os.makedirs(ckpt_save_path, exist_ok=True)
    # Create log file
    log_file_path = os.path.join(ckpt_save_path, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch\tLoss\tAccuracy\n")  # Write header
    
    # Training loop
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0

        hidden = None
        for batch in dataloader:
            # Separate state_tensor and hum_feature
            state_tensor = batch["state_tensor"].to(device)  # shape: (batch_size, 12, 16, 16)
            hum_features = batch["hum_feature"].to(device)   # shape: (batch_size, 4)
            
            # Initialize hidden state
            if hidden is None:
                hidden = cdc_model.init_hidden(batch_size)
            
            # Forward propagation (passing hum_features)
            accuracy, nce_loss, hidden = cdc_model(state_tensor, hum_features, hidden.detach())

            # Accumulate metrics
            total_loss += nce_loss.item()
            total_accuracy += accuracy
            batch_count += 1

            # Backpropagation
            optimizer.zero_grad()
            nce_loss.backward()
            optimizer.step()

        avg_loss = total_loss / batch_count
        avg_acc = total_accuracy / batch_count
        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")

        # Write to log file
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{epoch + 1}\t{avg_loss:.4f}\t{avg_acc:.2%}\n")

        # Update learning rate and save model
        scheduler.step()
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'cdc_model_state_dict': cdc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }, os.path.join(ckpt_save_path, f"checkpoint_epoch_{epoch}.pt"))

    
    print("Training is complete, and the model has been saved.")
    
# Main process
if __name__ == "__main__":
    train_sequential_model(
        ckpt_save_path=os.path.join('xcmd/PPOGC'),
        batch_size=50,
        n_epochs=100,
        # lr=3e-4,
        lr=3e-4 * math.sqrt(5),
        gamma_steplr=0.5,
        device_num='0',
        checkpoint_path=''
    )