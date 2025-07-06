from typing import Optional
import torch
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
        
        # Load or create BART model
        if self.custom_config is not None:
            # Create new model with custom config (randomly initialized)
            print("   Creating new BART model with random initialization...")
            self.model = BartModel(config=self.custom_config)
        else:
            # Load pretrained model
            self.model = BartModel.from_pretrained(self.model_name, config=self.config)

        self.latent_dim = self.config.d_model
        self.vocab_size = self.config.vocab_size
        
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
            
        self.report_trainable_parameters()
        
    def get_latent_dim(self) -> int:
        """Get the latent dimension of the encoder."""
        return self.latent_dim
    
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
        
        # Get raw token embeddings WITHOUT positional encodings
        # This gives us pure content embeddings as clean latents
        token_embeddings = self.model.get_encoder().embed_tokens(input_ids)
        
        return token_embeddings

    def get_vocab_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to vocabulary logits using weight tying.
        
        This implements weight tying by using the same weight matrix for both
        embedding lookup and output projection.
        
        Args:
            hidden_states: [B, L, C] hidden representations
            
        Returns:
            logits: [B, L, vocab_size] vocabulary logits
        """
        # Weight tying: use embedding weights as output projection
        # This is the key mechanism that prevents embedding collapse
        embed_weight = self.model.get_encoder().embed_tokens.weight
        assert isinstance(embed_weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return F.linear(hidden_states, embed_weight)
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Get the embedding weight matrix for clamping.
        
        Returns:
            Embedding weights [vocab_size, latent_dim]
        """
        return self.model.get_encoder().embed_tokens.weight
        
    def forward_encoder(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BART encoder.
        
        Args:
            inputs_embeds: Input embeddings [B, L, C]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Encoder output [B, L, C]
        """
        # Process through BART encoder
        encoder_outputs = self.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        return encoder_outputs.last_hidden_state
    
    def report_trainable_parameters(self):
        """Report the number of trainable parameters."""
        encoder = self.model.encoder
        
        embedding_params = sum(p.numel() for p in encoder.embed_tokens.parameters()) + \
                          sum(p.numel() for p in self.embed_positions.parameters()) + \
                          sum(p.numel() for p in self.layernorm_embedding.parameters())
        print(f"✅ Made BART embedding layers trainable ({embedding_params:,} trainable params)")
        
        transformer_params = sum(p.numel() for p in self.transformer_layers.parameters() if p.requires_grad)
        print(f"✅ Made BART transformer layers trainable ({transformer_params:,} trainable params)") 