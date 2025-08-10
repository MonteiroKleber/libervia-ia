"""
libervia-ia-Core v1 - Attention Mechanism
Implementação otimizada de Multi-Head Attention com Flash Attention e Sliding Window
"""

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    KVCache,
    _make_causal_mask,
    _expand_mask,
)

# Importações opcionais para Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None


class LiberviaAttention(nn.Module):
    """
    Multi-Head Attention otimizado para libervia-ia.
    
    Características:
    - Flash Attention v2 (quando disponível)
    - Sliding Window attention para sequências longas
    - RoPE (Rotary Position Embedding)
    - KV-Cache para inferência eficiente
    - Suporte a gradient checkpointing
    
    Args:
        config: Configuração do modelo
        layer_idx: Índice da camada (para debugging)
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        # Dimensões
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.n_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Sliding window
        self.sliding_window = config.sliding_window
        self.use_sliding_window = config.use_sliding_window and self.sliding_window is not None
        
        # Verificações
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) deve ser divisível por num_heads ({self.num_heads})"
            )
        
        # Projeções lineares (sem bias por padrão)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_sequence_length,
            base=config.rope_base,
        )
        
        # Dropout
        self.attention_dropout = config.attention_dropout
        self.dropout = nn.Dropout(self.attention_dropout) if self.attention_dropout > 0.0 else None
        
        # Flag para usar Flash Attention
        self.use_flash_attention = (
            config.use_flash_attention and 
            FLASH_ATTENTION_AVAILABLE and 
            torch.cuda.is_available()
        )
        
        if config.use_flash_attention and not self.use_flash_attention:
            warnings.warn(
                "Flash Attention foi solicitado mas não está disponível. "
                "Usando atenção manual. Para usar Flash Attention, instale: "
                "pip install flash-attn --no-build-isolation"
            )
    
    def _reshape_for_attention(
        self, 
        tensor: torch.Tensor, 
        num_heads: int
    ) -> torch.Tensor:
        """
        Reshape tensor para formato de atenção.
        
        Args:
            tensor: [batch_size, seq_len, num_heads * head_dim]
            num_heads: Número de cabeças
            
        Returns:
            Tensor: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repete key/value heads para Grouped Query Attention.
        
        Args:
            hidden_states: [batch, num_key_value_heads, seq_len, head_dim]
            n_rep: Número de repetições
            
        Returns:
            Tensor repetido [batch, num_heads, seq_len, head_dim]
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass usando Flash Attention.
        
        Args:
            query_states: [batch, num_heads, seq_len, head_dim]
            key_states: [batch, num_heads, seq_len, head_dim]  
            value_states: [batch, num_heads, seq_len, head_dim]
            attention_mask: Máscara de atenção (opcional)
            is_causal: Se deve usar máscara causal
            
        Returns:
            Output da atenção [batch, num_heads, seq_len, head_dim]
        """
        # Flash Attention espera [batch, seq_len, num_heads, head_dim]
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        
        # Parâmetros para Flash Attention
        dropout_p = self.attention_dropout if self.training else 0.0
        
        # Window size para sliding window
        window_size = (-1, -1)  # Sem limite por padrão
        if self.use_sliding_window and self.sliding_window:
            window_size = (self.sliding_window - 1, 0)
        
        # Aplicar Flash Attention
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=None,  # Usar padrão: 1/sqrt(head_dim)
            causal=is_causal,
            window_size=window_size,
            return_attn_probs=False
        )
        
        # Voltar para formato [batch, num_heads, seq_len, head_dim]
        return attn_output.transpose(1, 2)
    
    def _manual_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass usando atenção manual otimizada.
        
        Args:
            query_states: [batch, num_heads, seq_len, head_dim]
            key_states: [batch, num_heads, kv_seq_len, head_dim]
            value_states: [batch, num_heads, kv_seq_len, head_dim]
            attention_mask: Máscara de atenção
            is_causal: Se deve usar máscara causal
            
        Returns:
            Output da atenção [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        
        # Calcular scores de atenção
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        
        # Escalar por sqrt(head_dim)
        attn_weights = attn_weights / math.sqrt(head_dim)
        
        # Aplicar sliding window se configurado
        if self.use_sliding_window and self.sliding_window and kv_seq_len > self.sliding_window:
            # Criar máscara de sliding window
            sliding_mask = torch.triu(
                torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=query_states.device),
                diagonal=kv_seq_len - q_len - self.sliding_window + 1
            )
            sliding_mask = sliding_mask | torch.tril(
                torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=query_states.device),
                diagonal=kv_seq_len - q_len
            )
            attn_weights = attn_weights.masked_fill(~sliding_mask, torch.finfo(attn_weights.dtype).min)
        
        # Aplicar máscara causal
        if is_causal and q_len > 1:
            causal_mask = torch.triu(
                torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=query_states.device),
                diagonal=kv_seq_len - q_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, torch.finfo(attn_weights.dtype).min)
        
        # Aplicar máscara adicional se fornecida
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Dropout
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Aplicar atenção aos values
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[torch.Tensor]]:
        """
        Forward pass da atenção.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Máscara de atenção [batch_size, seq_len]
            position_ids: IDs de posição [batch_size, seq_len]
            past_key_value: Cache de KV anterior
            output_attentions: Se deve retornar pesos de atenção
            use_cache: Se deve usar/atualizar cache
            cache_position: Posição no cache
            
        Returns:
            Tuple (attn_output, present_key_value, attn_weights)
        """
        batch_size, q_len, _ = hidden_states.size()
        
        # Projeções Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape para atenção multi-head
        query_states = self._reshape_for_attention(query_states, self.num_heads)
        key_states = self._reshape_for_attention(key_states, self.num_key_value_heads)
        value_states = self._reshape_for_attention(value_states, self.num_key_value_heads)
        
        # Aplicar RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Gerenciar KV cache
        past_key_values_length = 0
        if past_key_value is not None:
            past_key_values_length = past_key_value.cache_position
            if cache_position is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, cache_position[0])
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, past_key_values_length)
        
        # Repetir KV heads se necessário (Grouped Query Attention)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Determinar se é causal
        is_causal = attention_mask is None and q_len > 1
        
        # Escolher implementação de atenção
        if self.use_flash_attention and not output_attentions:
            # Usar Flash Attention
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, is_causal
            )
            attn_weights = None
        else:
            # Usar atenção manual
            attn_output = self._manual_attention_forward(
                query_states, key_states, value_states, attention_mask, is_causal
            )
            attn_weights = None  # Para economia de memória, não retornar por padrão
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)
        
        # Projeção final
        attn_output = self.o_proj(attn_output)
        
        # Preparar outputs
        present_key_value = past_key_value if use_cache else None
        
        return attn_output, present_key_value, attn_weights


class LiberviaFlashAttention2(LiberviaAttention):
    """
    Versão especializada usando apenas Flash Attention v2.
    
    Para casos onde queremos garantir o uso de Flash Attention,
    com fallback para atenção manual se não disponível.
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None) -> None:
        super().__init__(config, layer_idx)
        
        # Forçar uso de Flash Attention se disponível
        self.use_flash_attention = FLASH_ATTENTION_AVAILABLE
        
        if not self.use_flash_attention:
            warnings.warn(
                "Flash Attention v2 não está disponível. Usando atenção manual. "
                "Para melhor performance, instale: pip install flash-attn --no-build-isolation"
            )


class LiberviaMultiHeadAttention(nn.Module):
    """
    Wrapper para diferentes tipos de atenção.
    
    Permite trocar facilmente entre implementações de atenção
    baseado na configuração.
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        
        # Escolher implementação baseada na config
        attn_implementation = getattr(config, 'attn_implementation', 'auto')
        
        if attn_implementation == 'flash_attention_2':
            self.attention = LiberviaFlashAttention2(config, layer_idx)
        elif attn_implementation == 'manual' or not FLASH_ATTENTION_AVAILABLE:
            self.attention = LiberviaAttention(config, layer_idx)
        else:  # 'auto'
            self.attention = LiberviaAttention(config, layer_idx)
    
    def forward(self, *args, **kwargs):
        """Delegete para implementação específica."""
        return self.attention(*args, **kwargs)


# Função factory para criar atenção
def create_attention_layer(config, layer_idx: Optional[int] = None) -> nn.Module:
    """
    Factory function para criar camada de atenção apropriada.
    
    Args:
        config: Configuração do modelo
        layer_idx: Índice da camada
        
    Returns:
        Instância da camada de atenção
    """
    return LiberviaMultiHeadAttention(config, layer_idx)


# Funções utilitárias para diagnóstico
def check_flash_attention_availability() -> dict:
    """
    Verifica disponibilidade e versão do Flash Attention.
    
    Returns:
        Dict com informações sobre Flash Attention
    """
    info = {
        'available': FLASH_ATTENTION_AVAILABLE,
        'cuda_available': torch.cuda.is_available(),
        'version': None,
        'recommendation': None
    }
    
    if FLASH_ATTENTION_AVAILABLE:
        try:
            import flash_attn
            info['version'] = getattr(flash_attn, '__version__', 'unknown')
        except:
            pass
    
    if not FLASH_ATTENTION_AVAILABLE:
        if torch.cuda.is_available():
            info['recommendation'] = "pip install flash-attn --no-build-isolation"
        else:
            info['recommendation'] = "Flash Attention requer CUDA. Usando atenção manual."
    
    return info


if __name__ == "__main__":
    # Teste básico
    from ..config import LiberviaConfig
    
    config = LiberviaConfig.from_model_size("2b")
    attention = create_attention_layer(config, layer_idx=0)
    
    print(f"Atenção criada: {type(attention.attention).__name__}")
    print(f"Flash Attention: {check_flash_attention_availability()}")
    
    # Teste com tensor dummy
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        output, _, _ = attention(hidden_states)
        print(f"Output shape: {output.shape}")
        print("✅ Teste básico passou!")