"""
libervia-ia-Core v1 - Base Components
Componentes fundamentais para o modelo Transformer
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Mais estável que LayerNorm tradicional, usado em LLaMA e outros modelos modernos.
    Normaliza apenas pela magnitude, sem recentragem (sem bias).
    
    Args:
        hidden_size: Dimensão dos embeddings
        eps: Epsilon para estabilidade numérica
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do RMSNorm.
        
        Args:
            hidden_states: Tensor de entrada [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor normalizado com mesma forma
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Calcular variância (média dos quadrados)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Codifica informação posicional rotacionando vetores de query e key
    no espaço complexo. Permite extrapolação para sequências mais longas.
    
    Args:
        dim: Dimensão da cabeça de atenção  
        max_position_embeddings: Comprimento máximo de sequência
        base: Base para frequências (default: 10000)
        device: Device onde criar os tensors
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calcular frequências para RoPE
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache para cos/sin pre-computados
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=device, 
            dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(
        self, 
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> None:
        """Pre-computa cache de cos/sin para eficiência."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.outer(t, self.inv_freq)
        # Concatenar para obter [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do RoPE.
        
        Args:
            x: Tensor para determinar device/dtype [batch, heads, seq_len, head_dim]
            seq_len: Comprimento da sequência (opcional)
            position_ids: IDs de posição específicos (opcional)
            
        Returns:
            Tuple (cos, sin) para aplicar rotação
        """
        # Usar comprimento da sequência do tensor se não fornecido
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Atualizar cache se necessário
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        if position_ids is None:
            # Usar posições sequenciais padrão
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            # Usar posições específicas
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rota metade dos embeddings.
    
    Reorganiza tensor [batch, heads, seq_len, head_dim] dividindo head_dim ao meio
    e aplicando rotação [-x2, x1] para implementar multiplicação complexa.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor, 
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica Rotary Position Embedding aos tensores query e key.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim] 
        cos: Cosenos pre-computados
        sin: Senos pre-computados
        position_ids: Posições específicas (opcional)
        unsqueeze_dim: Dimensão para unsqueeze (default: 1 para heads)
        
    Returns:
        Tuple (q_embed, k_embed) com RoPE aplicado
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    
    Função de ativação usada no FFN, combina Swish (SiLU) com gating:
    SwiGLU(x) = Swish(xW + b) ⊙ (xV + c)
    
    Mais eficiente que ReLU para modelos de linguagem.
    
    Args:
        dim: Dimensão de entrada
        hidden_dim: Dimensão oculta do FFN
        bias: Se deve usar bias (default: False)
    """
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do SwiGLU.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, dim]
            
        Returns:
            Tensor processado [batch_size, seq_len, dim]
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU: swish(gate) * up
        swish_gate = F.silu(gate)  # SiLU é equivalente ao Swish
        gated = swish_gate * up
        
        return self.down_proj(gated)


class LiberviaFFN(nn.Module):
    """
    Feed-Forward Network para libervia-ia.
    
    Implementa FFN com SwiGLU activation e otimizações:
    - Hidden dimension é múltiplo de 64 para otimização CUDA
    - Sem bias por padrão para economia de parâmetros
    - Suporte a gradient checkpointing
    
    Args:
        config: Configuração do modelo
    """
    
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        
        # Calcular dimensão oculta
        hidden_dim = int(config.d_model * config.ffn_mult)
        # Garantir múltiplo de 64 para otimização CUDA
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.hidden_size = config.d_model
        self.intermediate_size = hidden_dim
        
        # SwiGLU FFN
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Dropout se configurado
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do FFN.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor processado [batch_size, seq_len, hidden_size]
        """
        # SwiGLU
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = F.silu(gate) * up
        
        # Projeção final
        output = self.down_proj(intermediate)
        
        # Dropout se configurado
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output


class KVCache:
    """
    Key-Value Cache para inferência eficiente.
    
    Armazena keys e values computados anteriormente para evitar
    recomputação durante geração sequencial.
    
    Args:
        max_batch_size: Tamanho máximo do batch
        max_seq_len: Comprimento máximo de sequência  
        n_heads: Número de cabeças de atenção
        head_dim: Dimensão de cada cabeça
        device: Device onde armazenar o cache
        dtype: Tipo de dados
    """
    
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Pre-alocar cache para eficiência
        self.k_cache = torch.zeros(
            max_batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            max_batch_size, n_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        
        # Rastrear posição atual
        self.cache_position = 0
    
    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Atualiza cache com novos keys e values.
        
        Args:
            keys: Novos keys [batch_size, n_heads, seq_len, head_dim]
            values: Novos values [batch_size, n_heads, seq_len, head_dim]
            start_pos: Posição inicial na sequência
            
        Returns:
            Tuple (all_keys, all_values) incluindo histórico
        """
        batch_size, n_heads, seq_len, head_dim = keys.shape
        
        # Verificar limites
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(f"Sequência muito longa: {start_pos + seq_len} > {self.max_seq_len}")
        
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch muito grande: {batch_size} > {self.max_batch_size}")
        
        # Atualizar cache in-place
        self.k_cache[:batch_size, :, start_pos:start_pos + seq_len] = keys
        self.v_cache[:batch_size, :, start_pos:start_pos + seq_len] = values
        
        # Retornar visão do cache até a posição atual
        end_pos = start_pos + seq_len
        return (
            self.k_cache[:batch_size, :, :end_pos],
            self.v_cache[:batch_size, :, :end_pos]
        )
    
    def get(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtém keys e values do cache.
        
        Args:
            batch_size: Tamanho do batch
            seq_len: Comprimento da sequência
            
        Returns:
            Tuple (keys, values) do cache
        """
        return (
            self.k_cache[:batch_size, :, :seq_len],
            self.v_cache[:batch_size, :, :seq_len]
        )
    
    def clear(self) -> None:
        """Limpa o cache zerando todos os valores."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_position = 0


# Funções utilitárias para otimização

def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """
    Cria máscara causal para atenção.
    
    Args:
        input_ids_shape: Forma do tensor de entrada (batch_size, seq_len)
        dtype: Tipo de dados da máscara
        device: Device onde criar a máscara
        past_key_values_length: Comprimento do cache anterior
        
    Returns:
        Máscara causal [seq_len, seq_len + past_length]
    """
    batch_size, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """
    Expande máscara de atenção para formato correto.
    
    Args:
        mask: Máscara original [batch_size, src_len]
        dtype: Tipo de dados de saída
        tgt_len: Comprimento do target (default: mesmo que source)
        
    Returns:
        Máscara expandida [batch_size, 1, tgt_len, src_len]
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)
    
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)