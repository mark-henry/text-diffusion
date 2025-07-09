from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel, BartConfig
from .base_encoder import BaseEncoder


class BartEncoder(BaseEncoder):
    """
    BART encoder implementation for diffusion language models.
    
    Encapsulates BART-specific functionality and provides a unified interface
    through the BaseEncoder abstract class.
    """
    
    def __init__(self, model_name: str = "facebook/bart-base", max_length: int = 64, 
                 dropout: float = 0.1, custom_config: Optional[BartConfig] = None):
        super().__init__(model_name, max_length, dropout, custom_config)
        self.load_model()
        
    def load_model(self):
        """Load and initialize the BART model."""
        # Load BART model and extract components
        if self.custom_config is not None:
            self.config = self.custom_config
        else:
            self.config = BartConfig.from_pretrained(self.model_name)
        
        # Configure BART's dropout to match our dropout setting
        self.config.hidden_dropout_prob = self.dropout
        self.config.attention_probs_dropout_prob = self.dropout
        
        # Extract dimensions
        self.embedding_dim = getattr(self.config, 'embedding_dim', self.config.d_model)  # Fallback to d_model if not specified
        self.transformer_hidden_dim = self.config.d_model  # Transformer hidden dimension
        self.vocab_size = self.config.vocab_size
        
        # Load or create BART model
        if self.custom_config is not None:
            # Create new model with custom config (randomly initialized)
            print("   Creating new BART model with random initialization...")
            self.model = BartModel(config=self.custom_config)
        else:
            # Load pretrained model
            self.model = BartModel.from_pretrained(self.model_name, config=self.config)
        
        # Ensure max_length doesn't exceed BART's position embedding limit
        max_pos_embeds = self.config.max_position_embeddings
        if self.max_length > max_pos_embeds:
            print(f"Warning: max_length {self.max_length} exceeds BART's max_position_embeddings {max_pos_embeds}")
            self.max_length = min(self.max_length, max_pos_embeds - 2)  # -2 for safety margin
            print(f"Adjusted max_length to {self.max_length}")
        
        # Extract BART components
        encoder = self.model.encoder
        # Remove decoder weights to save memory - we only need encoder
        del self.model.decoder
        
        # TRAINABLE: BART's embedding layers (now learnable)
        self.embed_positions = encoder.embed_positions  
        self.layernorm_embedding = encoder.layernorm_embedding
        
        # Unfreeze embedding layers - make them trainable
        for param in encoder.embed_tokens.parameters():
            param.requires_grad = True
        for param in self.embed_positions.parameters():
            param.requires_grad = True
        for param in self.layernorm_embedding.parameters():
            param.requires_grad = True
        
        # TRAINABLE: unfreeze BART's transformer layers
        self.transformer_layers = encoder.layers
        for param in self.transformer_layers.parameters():
            param.requires_grad = True
        
        # Input up-projection: embedding space -> transformer hidden space
        self.input_up_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.transformer_hidden_dim),
            nn.Tanh(), 
            nn.Linear(self.transformer_hidden_dim, self.transformer_hidden_dim)
        )
        
        # Output down-projection: transformer hidden space -> embedding space
        self.output_down_proj = nn.Sequential(
            nn.Linear(self.transformer_hidden_dim, self.transformer_hidden_dim),
            nn.Tanh(), 
            nn.Linear(self.transformer_hidden_dim, self.embedding_dim)
        )
        
        # Always use custom token embeddings for simplicity and consistency
        print(f"   Creating custom token embeddings with embedding_dim={self.embedding_dim}")
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # Initialize with similar distribution to BART's embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
            
        self.report_trainable_parameters()
        
    def get_latent_dim(self) -> int:
        """Get the latent dimension of the encoder (transformer hidden dimension)."""
        return self.transformer_hidden_dim
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension (before projection to transformer hidden size)."""
        return self.embedding_dim
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def get_max_position_embeddings(self) -> int:
        """Get maximum position embeddings supported."""
        return self.config.max_position_embeddings
    
    def get_pad_token_id(self) -> int:
        """Get the pad token ID."""
        return self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 1
        
    def embed_tokens(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute embeddings from token IDs using BART embeddings.
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length doesn't exceed BART's limits
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        token_embeddings = self.token_embeddings(input_ids)
        
        return token_embeddings

    def get_vocab_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to vocabulary logits. Same weights are used for both embedding lookup and output projection.
        
        Args:
            hidden_states: [B, L, embedding_dim] hidden representations in embedding space
            
        Returns:
            logits: [B, L, vocab_size] vocabulary logits
        """
        embed_weight = self.token_embeddings.weight
        
        assert isinstance(embed_weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return F.linear(hidden_states, embed_weight)
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Get the embedding weight matrix for clamping.
        
        Returns:
            Embedding weights [vocab_size, embedding_dim]
        """
        weight = self.token_embeddings.weight
        
        assert isinstance(weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return weight
        
    def forward_encoder(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BART encoder with projections.
        
        Args:
            inputs_embeds: Input embeddings [B, L, embedding_dim] in embedding space
            attention_mask: Attention mask [B, L]
            
        Returns:
            Encoder output [B, L, embedding_dim] projected back to embedding space
        """
        # Project from embedding space to transformer hidden space
        projected_inputs = self.input_up_proj(inputs_embeds)  # [B, L, d_model]
        
        # Process through BART encoder
        encoder_outputs = self.model.encoder(
            inputs_embeds=projected_inputs,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Project back from transformer hidden space to embedding space
        hidden_states = encoder_outputs.last_hidden_state  # [B, L, d_model]
        output_embeddings = self.output_down_proj(hidden_states)  # [B, L, embedding_dim]
        
        return output_embeddings
    
    def report_trainable_parameters(self):
        """Report the number of trainable parameters."""
        encoder = self.model.encoder
        
        # Count embedding parameters
        embedding_params = sum(p.numel() for p in self.token_embeddings.parameters())
        print(f"✅ Custom token embeddings trainable ({embedding_params:,} trainable params)")
        
        # Count positional embeddings and layer norm
        pos_embed_params = sum(p.numel() for p in self.embed_positions.parameters())
        layer_norm_params = sum(p.numel() for p in self.layernorm_embedding.parameters())
        print(f"✅ BART positional embeddings and layer norm trainable ({pos_embed_params + layer_norm_params:,} trainable params)")
        
        # Count projection layer parameters
        input_proj_params = sum(p.numel() for p in self.input_up_proj.parameters() if p.requires_grad)
        output_proj_params = sum(p.numel() for p in self.output_down_proj.parameters() if p.requires_grad)
        total_projections = input_proj_params + output_proj_params
        print(f"✅ Projection layers trainable ({total_projections:,} trainable params)")
        print(f"   - Input up-projection: {input_proj_params:,}")
        print(f"   - Output down-projection: {output_proj_params:,}")
        
        # Count transformer parameters
        transformer_params = sum(p.numel() for p in self.transformer_layers.parameters() if p.requires_grad)
        print(f"✅ BART transformer layers trainable ({transformer_params:,} trainable params)") 