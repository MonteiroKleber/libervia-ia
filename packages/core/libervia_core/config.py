"""
libervia-ia-Core v1 - Model Configuration
Configuração serializável e type-safe para modelos Transformer
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal

import torch


@dataclass
class LiberviaConfig:
    """
    Configuração do modelo libervia-ia-Core.
    
    Esta classe define todos os hiperparâmetros arquiteturais e de treinamento
    do modelo Transformer decoder-only, seguindo as melhores práticas modernas.
    
    Args:
        model_name: Nome identificador do modelo
        d_model: Dimensão do modelo (embedding size)
        n_layers: Número de camadas transformer
        n_heads: Número de cabeças de atenção
        head_dim: Dimensão de cada cabeça (d_model / n_heads)
        ffn_mult: Multiplicador para dimensão do FFN (tipicamente 2.67)
        vocab_size: Tamanho do vocabulário
        max_sequence_length: Comprimento máximo de sequência
        sliding_window: Tamanho da janela deslizante de atenção
        rope_base: Base para Rotary Position Embedding
        norm_eps: Epsilon para RMSNorm
        use_flash_attention: Se deve usar Flash Attention v2
        use_kv_cache: Se deve usar KV-Cache para inferência
        tie_word_embeddings: Se deve compartilhar pesos embedding/output
        dropout: Taxa de dropout (0.0 para inferência)
        gradient_checkpointing: Se deve usar gradient checkpointing
        torch_dtype: Tipo de dados PyTorch para o modelo
    """
    
    # Identificação
    model_name: str = "libervia-core"
    model_type: str = "libervia"
    version: str = "1.0.0"
    
    # Arquitetura Principal
    d_model: int = 2048
    n_layers: int = 22
    n_heads: int = 16
    head_dim: Optional[int] = None  # Auto-calculado como d_model / n_heads
    ffn_mult: float = 2.67
    vocab_size: int = 32000
    
    # Contexto e Posição
    max_sequence_length: int = 4096
    sliding_window: Optional[int] = None  # None = usar max_sequence_length
    rope_base: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Normalização
    norm_eps: float = 1e-6
    norm_type: str = "rmsnorm"  # "rmsnorm" ou "layernorm"
    
    # Otimizações
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    use_sliding_window: bool = True
    tie_word_embeddings: bool = True
    scale_attn_weights: bool = True
    
    # Treinamento
    dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    gradient_checkpointing: bool = False
    
    # Quantização
    quantization_config: Optional[Dict[str, Any]] = None
    
    # Tipos de dados
    torch_dtype: str = "float16"  # "float32", "float16", "bfloat16"
    
    # Metadados
    total_params: Optional[int] = field(default=None, init=False)
    memory_footprint_mb: Optional[float] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """Validação e auto-preenchimento após inicialização."""
        # Auto-calcular head_dim se não fornecido
        if self.head_dim is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) deve ser divisível por n_heads ({self.n_heads})"
                )
            self.head_dim = self.d_model // self.n_heads
        
        # Verificar consistência head_dim
        expected_head_dim = self.d_model // self.n_heads
        if self.head_dim != expected_head_dim:
            raise ValueError(
                f"head_dim ({self.head_dim}) inconsistente com d_model/n_heads ({expected_head_dim})"
            )
        
        # Auto-configurar sliding_window
        if self.sliding_window is None:
            self.sliding_window = self.max_sequence_length
        
        # Calcular parâmetros totais
        self.total_params = self.estimate_parameters()
        self.memory_footprint_mb = self.estimate_memory_footprint()
        
        # Validações
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validações de consistência da configuração."""
        if self.d_model <= 0:
            raise ValueError("d_model deve ser positivo")
        
        if self.n_layers <= 0:
            raise ValueError("n_layers deve ser positivo")
        
        if self.n_heads <= 0:
            raise ValueError("n_heads deve ser positivo")
        
        if self.vocab_size <= 0:
            raise ValueError("vocab_size deve ser positivo")
        
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length deve ser positivo")
        
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout deve estar entre 0.0 e 1.0")
        
        if self.torch_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError("torch_dtype deve ser 'float32', 'float16' ou 'bfloat16'")
    
    def estimate_parameters(self) -> int:
        """Estima número total de parâmetros do modelo."""
        # Embedding de tokens
        token_embed_params = self.vocab_size * self.d_model
        
        # Camadas transformer
        # Cada camada tem: attn (4 projeções) + ffn (3 projeções) + 2 norms
        hidden_dim = int(self.d_model * self.ffn_mult)
        # Garantir múltiplo de 64 para otimização
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        per_layer_params = (
            # Attention: q, k, v, o projections
            4 * (self.d_model * self.d_model) +
            # FFN: gate, up, down projections  
            3 * (self.d_model * hidden_dim) +
            # RMSNorm weights (2 per layer)
            2 * self.d_model
        )
        
        transformer_params = self.n_layers * per_layer_params
        
        # Norma final
        final_norm_params = self.d_model
        
        # Projeção de saída (compartilhada com embedding se tie_word_embeddings=True)
        if self.tie_word_embeddings:
            output_params = 0
        else:
            output_params = self.vocab_size * self.d_model
        
        total = token_embed_params + transformer_params + final_norm_params + output_params
        return total
    
    def estimate_memory_footprint(self) -> float:
        """Estima footprint de memória em MB para diferentes precisões."""
        if self.total_params is None:
            params = self.estimate_parameters()
        else:
            params = self.total_params
        
        # Bytes por parâmetro baseado no tipo
        bytes_per_param = {
            "float32": 4,
            "float16": 2, 
            "bfloat16": 2
        }
        
        bytes_per_param_value = bytes_per_param[self.torch_dtype]
        total_bytes = params * bytes_per_param_value
        
        # Converter para MB
        return total_bytes / (1024 * 1024)
    
    @classmethod
    def from_model_size(cls, size: str) -> LiberviaConfig:
        """
        Factory method para criar configurações baseadas no tamanho do modelo.
        
        Args:
            size: Tamanho do modelo ("2b", "4b", "7b")
            
        Returns:
            Configuração correspondente ao tamanho
        """
        size = size.lower()
        
        if size == "2b":
            return cls(
                model_name="libervia-core-2b",
                d_model=2048,
                n_layers=22,
                n_heads=16,
                ffn_mult=2.67,
                vocab_size=32000,
                max_sequence_length=4096,
                sliding_window=4096,
                torch_dtype="float16"
            )
        elif size == "4b":
            return cls(
                model_name="libervia-core-4b", 
                d_model=2560,
                n_layers=32,
                n_heads=20,
                ffn_mult=2.67,
                vocab_size=32000,
                max_sequence_length=8192,
                sliding_window=4096,
                torch_dtype="float16"
            )
        elif size == "7b":
            return cls(
                model_name="libervia-core-7b",
                d_model=4096,
                n_layers=32, 
                n_heads=32,
                ffn_mult=2.67,
                vocab_size=32000,
                max_sequence_length=8192,
                sliding_window=4096,
                torch_dtype="float16"
            )
        else:
            raise ValueError(f"Tamanho de modelo não suportado: {size}. Use '2b', '4b' ou '7b'")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: Union[str, Path]) -> LiberviaConfig:
        """
        Carrega configuração de um modelo pré-treinado.
        
        Args:
            model_name_or_path: Nome do modelo ou caminho para config.json
            
        Returns:
            Configuração carregada
        """
        if isinstance(model_name_or_path, str):
            model_name_or_path = Path(model_name_or_path)
        
        # Se é um diretório, procurar config.json
        if model_name_or_path.is_dir():
            config_path = model_name_or_path / "config.json"
        else:
            config_path = model_name_or_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuração não encontrada: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> LiberviaConfig:
        """
        Cria configuração a partir de dicionário.
        
        Args:
            config_dict: Dicionário com parâmetros de configuração
            
        Returns:
            Instância de configuração
        """
        # Filtrar apenas campos válidos
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte configuração para dicionário.
        
        Returns:
            Dicionário com todos os parâmetros
        """
        return asdict(self)
    
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Salva configuração em diretório.
        
        Args:
            save_directory: Diretório onde salvar config.json
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        config_path = save_directory / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_torch_dtype(self) -> torch.dtype:
        """
        Converte string de dtype para torch.dtype.
        
        Returns:
            Tipo de dados PyTorch correspondente
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map[self.torch_dtype]
    
    def update(self, **kwargs) -> LiberviaConfig:
        """
        Cria nova configuração com parâmetros atualizados.
        
        Args:
            **kwargs: Parâmetros a serem atualizados
            
        Returns:
            Nova instância de configuração
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def __str__(self) -> str:
        """Representação string legível da configuração."""
        return (
            f"LiberviaConfig(\n"
            f"  model_name='{self.model_name}',\n"
            f"  d_model={self.d_model},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_heads={self.n_heads},\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  max_sequence_length={self.max_sequence_length},\n"
            f"  total_params={self.total_params:,} ({self.total_params/1e9:.1f}B),\n"
            f"  memory_footprint={self.memory_footprint_mb:.1f}MB,\n"
            f"  torch_dtype='{self.torch_dtype}'\n"
            f")"
        )
    
    def __repr__(self) -> str:
        """Representação técnica da configuração."""
        return self.__str__()


# Configurações pré-definidas para acesso rápido
LIBERVIA_2B_CONFIG = LiberviaConfig.from_model_size("2b")
LIBERVIA_4B_CONFIG = LiberviaConfig.from_model_size("4b") 
LIBERVIA_7B_CONFIG = LiberviaConfig.from_model_size("7b")


def get_config_for_size(size: str) -> LiberviaConfig:
    """
    Função helper para obter configuração por tamanho.
    
    Args:
        size: Tamanho do modelo ("2b", "4b", "7b")
        
    Returns:
        Configuração correspondente
    """
    return LiberviaConfig.from_model_size(size)


if __name__ == "__main__":
    # Teste rápido das configurações
    for size in ["2b", "4b", "7b"]:
        config = LiberviaConfig.from_model_size(size)
        print(f"\n{config}")
        print(f"Parâmetros estimados: {config.total_params:,}")
        print(f"Memória estimada: {config.memory_footprint_mb:.1f} MB")