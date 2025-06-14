import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import wandb
import time
from typing import List, Tuple, Dict
import torch.nn.functional as F

class CosineNoiseScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create cosine schedule (better than linear)
        self.betas = self._cosine_beta_schedule(num_timesteps, beta_start, beta_end)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def _cosine_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)
        
    def add_noise(self, latents: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latents according to timesteps."""
        noise = torch.randn_like(latents)
        # Move alphas_cumprod to same device as timesteps
        alphas_cumprod = self.alphas_cumprod.to(timesteps.device)[timesteps]
        
        # Reshape for broadcasting with [B, C, L] format
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1)
        noisy_latents = torch.sqrt(alphas_cumprod) * latents + torch.sqrt(1 - alphas_cumprod) * noise
        return noisy_latents, noise

def pad_tensor(tensor: torch.Tensor, max_length: int) -> torch.Tensor:
    """Pad tensor to max_length."""
    current_length = tensor.shape[0]
    if current_length >= max_length:
        return tensor[:max_length]
    
    # Create padding
    padding = torch.zeros((max_length - current_length, tensor.shape[1]), dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=0)

class LatentDataset(Dataset):
    def __init__(self, model: BartForConditionalGeneration, tokenizer: BartTokenizer, 
                 dataset: List[str], max_length: int = 128):  # Even smaller for efficiency
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = [text for text in dataset if text.strip()]  # Filter empty texts
        self.max_length = max_length
        
        # Pre-compute statistics for normalization
        print("Computing dataset statistics for normalization...")
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for latent normalization."""
        sample_size = min(500, len(self.dataset))  # Smaller sample for efficiency
        latents_sample = []
        
        for i in range(0, sample_size, 20):  # Every 20th sample
            text = self.dataset[i]
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            with torch.no_grad():
                encoder_outputs = self.model.get_encoder()(inputs["input_ids"])
                latents = encoder_outputs.last_hidden_state.squeeze(0)
                latents = pad_tensor(latents, self.max_length)
                latents_sample.append(latents)
        
        if latents_sample:
            all_latents = torch.stack(latents_sample)
            self.latent_mean = all_latents.mean(dim=(0, 1), keepdim=True)
            self.latent_std = all_latents.std(dim=(0, 1), keepdim=True) + 1e-8
        else:
            self.latent_mean = torch.zeros(1, 768)
            self.latent_std = torch.ones(1, 768)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.dataset[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(inputs["input_ids"])
            latents = encoder_outputs.last_hidden_state.squeeze(0)
            latents = pad_tensor(latents, self.max_length)
            # Normalize latents
            latents = (latents - self.latent_mean.squeeze()) / self.latent_std.squeeze()
        return latents

class SimpleUNet1D(nn.Module):
    """Simplified UNet that actually works."""
    def __init__(self, in_channels=768, out_channels=768, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Simple encoder-decoder without skip connections for now
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        self.middle = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, out_channels, 3, padding=1),
        )
        
    def forward(self, x, timesteps):
        # Time embedding (not used in this simple version)
        t = timesteps.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_embed(t)
        
        # Simple forward pass
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        
        return x

def validate_model(model, val_loader, scheduler, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0
    val_cosine_sim = 0
    num_batches = 0
    
    with torch.no_grad():
        for latents in val_loader:
            latents = latents.to(device)
            batch_size = latents.shape[0]
            latents = latents.transpose(1, 2)  # [B, L, C] -> [B, C, L]
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents, timesteps)
            
            predicted_noise = model(noisy_latents, timesteps)
            loss = criterion(predicted_noise, noise)
            
            # Calculate cosine similarity
            noise_flat = noise.reshape(batch_size, -1)
            pred_flat = predicted_noise.reshape(batch_size, -1)
            cosine_sim = F.cosine_similarity(noise_flat, pred_flat, dim=1).mean()
            
            val_loss += loss.item()
            val_cosine_sim += cosine_sim.item()
            num_batches += 1
    
    return val_loss / num_batches, val_cosine_sim / num_batches

def train_denoiser(
    model: nn.Module,
    scheduler: CosineNoiseScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """Train the denoising model with validation."""
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_cosine_sim": [], "val_cosine_sim": []
    }
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({"model/total_parameters": total_params})
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_cosine_sim = 0
        epoch_start_time = time.time()
        
        for batch_idx, latents in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            latents = latents.to(device)
            batch_size = latents.shape[0]
            # latents should be [B, L, C] -> transpose to [B, C, L] for conv1d
            latents = latents.transpose(1, 2)
            
            timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
            noisy_latents, noise = scheduler.add_noise(latents, timesteps)
            
            predicted_noise = model(noisy_latents, timesteps)
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler_lr.step()
            
            # Calculate metrics
            noise_flat = noise.reshape(batch_size, -1)
            pred_flat = predicted_noise.reshape(batch_size, -1)
            cosine_sim = F.cosine_similarity(noise_flat, pred_flat, dim=1).mean()
            
            epoch_loss += loss.item()
            epoch_cosine_sim += cosine_sim.item()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_cosine_sim": cosine_sim.item(),
                    "train/learning_rate": scheduler_lr.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "train/batch": batch_idx,
                    "train/epoch": epoch,
                })
        
        # Validation
        val_loss, val_cosine_sim = validate_model(model, val_loader, scheduler, criterion, device)
        
        # Calculate epoch metrics
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_cosine_sim = epoch_cosine_sim / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_cosine_sim"].append(avg_train_cosine_sim)
        history["val_cosine_sim"].append(val_cosine_sim)
        
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": val_loss,
            "train/epoch_cosine_sim": avg_train_cosine_sim,
            "val/epoch_cosine_sim": val_cosine_sim,
            "train/epoch_time": epoch_time,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Cosine Sim: {avg_train_cosine_sim:.4f}, Val Cosine Sim: {val_cosine_sim:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_simple_denoiser.pt")
    
    return history

def main():
    # Initialize wandb
    wandb.init(
        project="text-diffusion",
        name="bart-denoiser-v3-simple",
        config={
            "model_type": "SimpleUNet1D",
            "encoder": "BART-base",
            "dataset": "WikiText-2",
            "batch_size": 32,  # Larger batch size
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "max_length": 128,  # Smaller for efficiency
            "noise_scheduler": "cosine",
            "num_timesteps": 1000,
            "optimizer": "AdamW",
            "lr_scheduler": "CosineAnnealing",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "normalization": "latent_normalization",
            "architecture": "simple_unet_no_skip_connections"
        }
    )
    
    # Load BART model and tokenizer
    print("Loading BART model and tokenizer...")
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Create dataset with normalization
    print("Creating latent dataset...")
    full_dataset = LatentDataset(bart_model, tokenizer, dataset['train']['text'], max_length=128)
    
    # Train/validation split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Log dataset info
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/val_size": len(val_dataset),
        "dataset/train_batches": len(train_loader),
        "dataset/val_batches": len(val_loader)
    })
    
    # Create simple model
    print("Creating simple UNet model...")
    model = SimpleUNet1D(in_channels=768, out_channels=768, hidden_dim=256)
    
    # Create noise scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=1000)
    
    # Train model
    print("Training denoising model...")
    start_time = time.time()
    history = train_denoiser(model, scheduler, train_loader, val_loader, num_epochs=2)
    total_time = time.time() - start_time
    
    # Log final metrics
    wandb.log({
        "training/total_time": total_time,
        "training/final_train_loss": history["train_loss"][-1],
        "training/final_val_loss": history["val_loss"][-1],
        "training/best_val_loss": min(history["val_loss"])
    })
    
    # Save final model
    torch.save(model.state_dict(), "final_simple_denoiser.pt")
    
    # Save models as wandb artifacts
    for model_name in ["best_simple_denoiser.pt", "final_simple_denoiser.pt"]:
        artifact = wandb.Artifact(f"simple-denoiser-{model_name.split('_')[0]}", type="model")
        artifact.add_file(model_name)
        wandb.log_artifact(artifact)
    
    print("Training complete!")
    wandb.finish()

if __name__ == "__main__":
    main() 