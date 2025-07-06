from typing import Optional
import torch
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from .base_encoder import BaseEncoder


class BertEncoder(BaseEncoder):
    """
    BERT encoder implementation for diffusion language models.
    
    Encapsulates BERT-specific functionality and provides a unified interface
    through the BaseEncoder abstract class.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 64, 
                 dropout: float = 0.1, custom_config: Optional[BertConfig] = None):
        super().__init__(model_name, max_length, dropout, custom_config)
        self.load_model()
        
    def load_model(self):
        """Load and initialize the BERT model."""
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
            self.model = BertModel(config=self.custom_config)
        else:
            # Load pretrained model
            self.model = BertModel.from_pretrained(self.model_name, config=self.config)

        self.latent_dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        
        # Ensure max_length doesn't exceed BERT's position embedding limit
        max_pos_embeds = self.config.max_position_embeddings
        if self.max_length > max_pos_embeds:
            print(f"Warning: max_length {self.max_length} exceeds BERT's max_position_embeddings {max_pos_embeds}")
            self.max_length = min(self.max_length, max_pos_embeds - 2)  # -2 for safety margin
            print(f"Adjusted max_length to {self.max_length}")
        
        # BERT is encoder-only, so we don't need to remove decoder like we did with BART
        # All BERT parameters are trainable by default
        
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
        Compute clean latents from token IDs using BERT encoder.
        This method gets the target embeddings that the diffusion model should predict.
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
        
        # Use BERT encoder directly with input_ids (this handles positional embeddings automatically)
        encoder_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Return the encoder's hidden state as our target clean latents
        return encoder_outputs.last_hidden_state

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
        embed_weight = self.model.embeddings.word_embeddings.weight
        assert isinstance(embed_weight, torch.Tensor), "Expected embedding weight to be a tensor"
        return F.linear(hidden_states, embed_weight)
    
    def get_embedding_weights(self) -> torch.Tensor:
        """
        Get the embedding weight matrix for clamping.
        
        Returns:
            Embedding weights [vocab_size, latent_dim]
        """
        full_vocab_embeddings = self.model.embeddings.word_embeddings.weight  # [full_vocab_size, embed_dim]
        model_vocab_size = self.config.vocab_size
        vocab_embeddings = full_vocab_embeddings[:model_vocab_size]  # [model_vocab_size, embed_dim]
        return vocab_embeddings
        
    def forward_encoder(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BERT encoder.
        
        Args:
            inputs_embeds: Input embeddings [B, L, C]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Encoder output [B, L, C]
        """
        # Process through BERT encoder using inputs_embeds
        encoder_outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        return encoder_outputs.last_hidden_state
    
    def report_trainable_parameters(self):
        """Report the number of trainable parameters."""
        # Count and report trainable parameters
        embedding_params = sum(p.numel() for p in self.model.embeddings.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        
        print(f"✅ BERT embedding layers trainable ({embedding_params:,} trainable params)")
        print(f"✅ BERT encoder layers trainable ({encoder_params:,} trainable params)") 