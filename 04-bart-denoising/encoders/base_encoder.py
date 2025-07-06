from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoder implementations used in diffusion language models.
    
    This class defines the interface that both BART and BERT encoders must implement
    to work with the unified DiffusionLM model.
    """
    
    def __init__(self, model_name: str, max_length: int, dropout: float, custom_config: Optional[Any] = None):
        """
        Initialize the encoder.
        
        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
            dropout: Dropout probability
            custom_config: Optional custom configuration
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.dropout = dropout
        self.custom_config = custom_config
        
        # These will be set by subclasses
        self.config = None
        self.model = None
        self.latent_dim = None
        self.vocab_size = None
        
    @abstractmethod
    def load_model(self):
        """Load and initialize the underlying model."""
        pass
    
    @abstractmethod
    def get_latent_dim(self) -> int:
        """Get the latent dimension of the encoder."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        pass
    
    @abstractmethod
    def get_max_position_embeddings(self) -> int:
        """Get maximum position embeddings supported."""
        pass
    
    @abstractmethod
    def get_pad_token_id(self) -> int:
        """Get the pad token ID."""
        pass
    
    @abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute clean latents from token IDs.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Clean latents [B, L, C]
        """
        pass
    
    @abstractmethod
    def get_vocab_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Convert hidden states to vocabulary logits using weight tying.
        
        Args:
            hidden_states: Hidden representations [B, L, C]
            
        Returns:
            Vocabulary logits [B, L, vocab_size]
        """
        pass
    
    @abstractmethod
    def forward_encoder(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            inputs_embeds: Input embeddings [B, L, C]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Encoder output [B, L, C]
        """
        pass
    
    @abstractmethod
    def report_trainable_parameters(self):
        """Report the number of trainable parameters."""
        pass
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Get the embedding weight matrix for clamping.
        
        Returns:
            Embedding weights [vocab_size, latent_dim]
        """
        # This calls the abstract get_vocab_logits indirectly through clamping
        # The specific implementation will be in subclasses
        raise NotImplementedError("Subclasses must implement get_embedding_weights") 