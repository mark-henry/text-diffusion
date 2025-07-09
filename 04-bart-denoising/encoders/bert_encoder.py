from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from .base_encoder import BaseEncoder


class BertEncoder(BaseEncoder):
    """
    BERT encoder implementation for diffusion language models.
    
    Follows the reference implementation by:
    1. Removing built-in BERT embeddings and pooler
    2. Using custom positional embeddings 
    3. Excluding [CLS] tokens from training targets
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 64, 
                 dropout: float = 0.1, custom_config: Optional[BertConfig] = None):
        super().__init__(model_name, max_length, dropout, custom_config)
        self.load_model()
        
    def load_model(self):
        """Load and initialize the BERT model following reference implementation."""
        # Load BERT model and extract components
        if self.custom_config is not None:
            self.config = self.custom_config
        else:
            self.config = BertConfig.from_pretrained(self.model_name)
        
        # Configure BERT's dropout to match our dropout setting
        self.config.hidden_dropout_prob = self.dropout
        self.config.attention_probs_dropout_prob = self.dropout
        
        # Load or create BERT model
        if self.custom_config is not None:
            # Create new model with custom config (randomly initialized)
            print("   Creating new BERT model with random initialization...")
            temp_bert = BertModel(config=self.custom_config)
        else:
            # Load pretrained model
            temp_bert = BertModel.from_pretrained(self.model_name, config=self.config)

        # Following reference implementation: delete built-in embeddings and pooler
        print("   Removing BERT's built-in embeddings and pooler...")
        del temp_bert.embeddings
        del temp_bert.pooler
        
        # Keep only the encoder layers
        self.encoder_layers = temp_bert.encoder
        
        # Create custom embedding layers following reference implementation
        print("   Creating custom embedding layers...")
        self.latent_dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        
        # Custom word embeddings (trainable)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.latent_dim)
        
        # Custom positional embeddings following reference implementation
        self.register_buffer("position_ids", torch.arange(self.config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.latent_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(self.latent_dim, eps=self.config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(self.dropout)
        
        # Ensure max_length doesn't exceed BERT's position embedding limit
        max_pos_embeds = self.config.max_position_embeddings
        if self.max_length > max_pos_embeds:
            print(f"Warning: max_length {self.max_length} exceeds BERT's max_position_embeddings {max_pos_embeds}")
            self.max_length = min(self.max_length, max_pos_embeds - 2)  # -2 for safety margin
            print(f"Adjusted max_length to {self.max_length}")
        
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
        return self.config.pad_token_id if hasattr(self.config, 'pad_token_id') else 0
        
    def embed_tokens(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute clean latents from token IDs using custom embeddings.
        
        Following reference implementation:
        1. Use custom word + positional embeddings
        2. Apply layer norm and dropout
        3. No special token handling needed (add_special_tokens=False used consistently)
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure sequence length doesn't exceed BERT's limits
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        # No need to check for [CLS] tokens since we use add_special_tokens=False consistently
        
        # Custom embedding computation following reference implementation
        # 1. Word embeddings
        word_embeds = self.word_embeddings(input_ids)
        
        # 2. Positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # 3. Combine embeddings
        embeddings = word_embeds + position_embeds
        
        # 4. Layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        return embeddings

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
        embed_weight = self.word_embeddings.weight
        assert isinstance(embed_weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return F.linear(hidden_states, embed_weight)
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Get the embedding weight matrix for clamping.
        
        Returns:
            Embedding weights [vocab_size, latent_dim]
        """
        return self.word_embeddings.weight
        
    def forward_encoder(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BERT encoder.
        
        Args:
            inputs_embeds: Input embeddings [B, L, C]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Encoder output [B, L, C]
        """
        # Convert attention mask to the format expected by BERT
        # BERT expects 1 for valid tokens, 0 for padding
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=inputs_embeds.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Process through BERT encoder layers
        encoder_outputs = self.encoder_layers(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        return encoder_outputs.last_hidden_state
    
    def report_trainable_parameters(self):
        """Report the number of trainable parameters."""
        # Count custom embedding parameters
        word_embed_params = sum(p.numel() for p in self.word_embeddings.parameters() if p.requires_grad)
        pos_embed_params = sum(p.numel() for p in self.position_embeddings.parameters() if p.requires_grad)
        layer_norm_params = sum(p.numel() for p in self.layer_norm.parameters() if p.requires_grad)
        
        # Count encoder parameters
        encoder_params = sum(p.numel() for p in self.encoder_layers.parameters() if p.requires_grad)
        
        total_custom_embed = word_embed_params + pos_embed_params + layer_norm_params
        print(f"✅ Custom embedding layers trainable ({total_custom_embed:,} trainable params)")
        print(f"   - Word embeddings: {word_embed_params:,}")
        print(f"   - Position embeddings: {pos_embed_params:,}")
        print(f"   - Layer norm: {layer_norm_params:,}")
        print(f"✅ BERT encoder layers trainable ({encoder_params:,} trainable params)") 