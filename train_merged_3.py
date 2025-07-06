import json
import os
import random
import data_tool1
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from modules import Xfeatmodel, hum, CPCmodel_PPOGCPM, CPCmodel_PPOGDI
import math

class SequentialChunkedDataset(IterableDataset):
    def __init__(self, folder_path, shuffle_files=False):
        self.folder_path = folder_path
        self.shuffle_files = shuffle_files
        self.json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if self.shuffle_files:
            random.shuffle(self.json_files)  # Files can be shuffled between files, but sequence within files remains fixed

    def process_file(self, json_file):
        """Process a single file and return the data (keep time series fixed)."""
        file_path = os.path.join(self.folder_path, json_file)
        data = data_tool1.load_and_fix_json(file_path)
        if data is None:
            return []
        # processed_data = data_tool.process_data(data)
        # Convert raw data into tensor format
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
        """Iterator: load data file by file."""
        for json_file in self.json_files:
            file_data = self.process_file(json_file)
            for item in file_data:
                yield item  # Return a dictionary containing state_tensor and hum_feature

def convert_state_to_tensor(data):
    converted_data = []
    for step_info in data:
        state = step_info["state"]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.permute(1, 0).view(12, 16, 16)
        converted_data.append({"step": step_info["step"], "state_tensor": state_tensor})
    return converted_data

def custom_collate(batch):
    state_tensors = [item["state_tensor"] for item in batch]
    hum_features = [item["hum_feature"] for item in batch]
    return {
        "state_tensor": torch.stack(state_tensors),
        "hum_feature": torch.stack(hum_features)
    }

def cosine_distance_loss(feat_new, feat_old1, feat_old2):
    """Calculate cosine distance loss between new features and old features.
    - Input shapes: feat_new, feat_old1, feat_old2 are all (batch_size, feature_dim)
    """

    # Calculate cosine similarity (range [-1, 1], 1 means same direction)
    cos_sim1 = F.cosine_similarity(feat_new, feat_old1, dim=1)  # Shape: (batch_size,)
    cos_sim2 = F.cosine_similarity(feat_new, feat_old2, dim=1)
    
    # Maximize distance: minimize the average of cosine similarities (make it close to -1)
    distance_loss = (cos_sim1.mean() + cos_sim2.mean()) / 2.0  # Combine losses from two old models
    return distance_loss


def train_sequential_model(ckpt_save_path, old_checkpoint_path1, old_checkpoint_path2, batch_size=100, n_epochs=10, lr=3e-4, gamma_steplr=0.5, device_num='0', checkpoint_path=None, lambda_div=0.5):
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    
    # Load two pre-trained models
    old_encoder1 = Xfeatmodel.XFeatModel().to(device)
    old_cpc_model1 = CPCmodel_PPOGCPM.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=old_encoder1).to(device)
    checkpoint_old1 = torch.load(old_checkpoint_path1, map_location=device)
    old_encoder1.load_state_dict(checkpoint_old1['encoder_state_dict'])
    old_cpc_model1.load_state_dict(checkpoint_old1['cdc_model_state_dict'])


    old_encoder2 = Xfeatmodel.XFeatModel().to(device)
    old_cpc_model2 = CPCmodel_PPOGDI.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=old_encoder2).to(device)
    checkpoint_old2 = torch.load(old_checkpoint_path2, map_location=device)
    old_encoder2.load_state_dict(checkpoint_old2['encoder_state_dict'])
    old_cpc_model2.load_state_dict(checkpoint_old2['cdc_model_state_dict'])

    
    # Freeze pre-trained models
    for model in [old_encoder1, old_cpc_model1]:    #, old_encoder2, old_cpc_model2]:
        for param in model.parameters():
            param.requires_grad = False

    # Initialize new model
    encoder = Xfeatmodel.XFeatModel().to(device)
    cdc_model = CPCmodel_PPOGCPM.CDCK5(timestep=12, batch_size=batch_size, seq_len=16, encoder=encoder).to(device)
    
    # Optimizer
    optimizer = Adam(list(encoder.parameters()), lr=lr)
    # scheduler = StepLR(optimizer, step_size=n_epochs // 10, gamma=gamma_steplr)

    # If checkpoint path is provided, load model state
    start_epoch = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        cdc_model.load_state_dict(checkpoint['cdc_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch}.")

    # Create data set
    dataset = SequentialChunkedDataset(folder_path="/mnt/671cbd8b-55cf-4eb4-af6d-a4ab48e8c9d2/JL/JL/passive", shuffle_files=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,  # Use custom collate function
        drop_last = True
    )
    
    # Create log file
    log_file_path = os.path.join(ckpt_save_path, "training_log.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure directory exists
    with open(log_file_path, "w") as f:
        f.write("Epoch\tNCE Loss\tCosine Dist Loss\tAccuracy\n")  # Write header

    for epoch in range(start_epoch, n_epochs):
        encoder.train()
        cdc_model.train()
        total_nce_loss = 0.0
        total_cosine_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            optimizer.zero_grad()
            state_tensor = batch["state_tensor"].to(device)
            hum_feature = batch["hum_feature"].to(device)

            # Forward pass
            output = cdc_model(state_tensor)
            nce_loss = cdc_model.loss(output, hum_feature)
            feat_new = encoder(state_tensor)
            feat_old1 = old_encoder1(state_tensor)
            feat_old2 = old_encoder2(state_tensor)
            cosine_loss = cosine_distance_loss(feat_new, feat_old1, feat_old2)

            # Combine losses
            loss = nce_loss + lambda_div * cosine_loss
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            total_nce_loss += nce_loss.item()
            total_cosine_loss += cosine_loss.item()
            total_correct += (output.argmax(dim=1) == hum_feature.argmax(dim=1)).sum().item()
            total_samples += output.size(0)

        # Calculate average loss and accuracy
        avg_nce_loss = total_nce_loss / len(dataloader)
        avg_cosine_loss = total_cosine_loss / len(dataloader)
        accuracy = total_correct / total_samples

        # Print and log results
        print(f"Epoch {epoch + 1}/{n_epochs}: NCE Loss: {avg_nce_loss:.4f}, Cosine Dist Loss: {avg_cosine_loss:.4f}, Accuracy: {accuracy:.4f}")
        with open(log_file_path, "a") as f:
            f.write(f"{epoch + 1}\t{avg_nce_loss:.4f}\t{avg_cosine_loss:.4f}\t{accuracy:.4f}\n")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'cdc_model_state_dict': cdc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(ckpt_save_path, f"epoch_{epoch + 1}.pt"))
