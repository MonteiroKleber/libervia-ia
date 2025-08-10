"""
libervia-ia-Core v1 - Core Package
Exports principais do pacote core
"""

from .config import (
    LiberviaConfig,
    LIBERVIA_2B_CONFIG,
    LIBERVIA_4B_CONFIG, 
    LIBERVIA_7B_CONFIG,
    get_config_for_size,
)

from .model import (
    LiberviaModel,
    LiberviaForCausalLM,
    LiberviaLMHeadModel,  # Alias
    LiberviaEmbedding,
    LiberviaDecoderLayer,
)

from .attention import (
    LiberviaAttention,
    LiberviaFlashAttention2,
    LiberviaMultiHeadAttention,
    create_attention_layer,
    check_flash_attention_availability,
)

from .components import (
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    LiberviaFFN,
    KVCache,
    apply_rotary_pos_emb,
    rotate_half,
)

# Versão do pacote
__version__ = "1.0.0"

# Exports principais
__all__ = [
    # Configuração
    "LiberviaConfig",
    "LIBERVIA_2B_CONFIG",
    "LIBERVIA_4B_CONFIG", 
    "LIBERVIA_7B_CONFIG",
    "get_config_for_size",
    
    # Modelo principal
    "LiberviaModel",
    "LiberviaForCausalLM",
    "LiberviaLMHeadModel",
    
    # Componentes de modelo
    "LiberviaEmbedding",
    "LiberviaDecoderLayer",
    
    # Atenção
    "LiberviaAttention",
    "LiberviaFlashAttention2", 
    "LiberviaMultiHeadAttention",
    "create_attention_layer",
    "check_flash_attention_availability",
    
    # Componentes base
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "LiberviaFFN",
    "KVCache",
    "apply_rotary_pos_emb",
    "rotate_half",
    
    # Versão
    "__version__",
]


def get_model_info():
    """
    Retorna informações sobre os modelos disponíveis.
    
    Returns:
        Dict com informações dos modelos
    """
    models = {}
    
    for size in ["2b", "4b", "7b"]:
        config = LiberviaConfig.from_model_size(size)
        models[size] = {
            "name": config.model_name,
            "parameters": config.total_params,
            "memory_mb": config.memory_footprint_mb,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "max_sequence_length": config.max_sequence_length,
        }
    
    return models


def print_model_info():
    """Imprime informações sobre os modelos disponíveis."""
    print("🌟 libervia-ia-Core v1 - Modelos Disponíveis")
    print("=" * 50)
    
    models = get_model_info()
    
    for size, info in models.items():
        print(f"\n📋 {info['name']}:")
        print(f"   Parâmetros: {info['parameters']:,} ({info['parameters']/1e9:.1f}B)")
        print(f"   Memória: {info['memory_mb']:.1f} MB")
        print(f"   Arquitetura: {info['d_model']}d × {info['n_layers']}L × {info['n_heads']}H")
        print(f"   Contexto: {info['max_sequence_length']:,} tokens")
    
    print(f"\n🔧 Flash Attention: {check_flash_attention_availability()}")
    print(f"📦 Versão: {__version__}")


# Função convenience para criar modelos rapidamente
def create_model(size: str = "2b", **kwargs) -> LiberviaForCausalLM:
    """
    Cria modelo libervia-ia rapidamente.
    
    Args:
        size: Tamanho do modelo ("2b", "4b", "7b")
        **kwargs: Argumentos adicionais para configuração
        
    Returns:
        Modelo pronto para uso
        
    Example:
        >>> model = create_model("2b")
        >>> model = create_model("4b", torch_dtype="bfloat16")
    """
    config = LiberviaConfig.from_model_size(size)
    
    # Aplicar kwargs à configuração
    if kwargs:
        config = config.update(**kwargs)
    
    return LiberviaForCausalLM(config)


def create_config(size: str = "2b", **kwargs) -> LiberviaConfig:
    """
    Cria configuração libervia-ia rapidamente.
    
    Args:
        size: Tamanho do modelo ("2b", "4b", "7b")
        **kwargs: Argumentos adicionais para configuração
        
    Returns:
        Configuração do modelo
        
    Example:
        >>> config = create_config("2b")
        >>> config = create_config("4b", max_sequence_length=16384)
    """
    config = LiberviaConfig.from_model_size(size)
    
    # Aplicar kwargs à configuração
    if kwargs:
        config = config.update(**kwargs)
    
    return config


# Verificação de sistema
def check_system_compatibility():
    """
    Verifica compatibilidade do sistema.
    
    Returns:
        Dict com informações de compatibilidade
    """
    import torch
    import sys
    
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "flash_attention": check_flash_attention_availability(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["cuda_memory_gb"] = props.total_memory / 1e9
        info["cuda_compute_capability"] = f"{props.major}.{props.minor}"
    
    return info


def print_system_info():
    """Imprime informações do sistema."""
    print("🖥️  System Compatibility Check")
    print("=" * 30)
    
    info = check_system_compatibility()
    
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"CUDA: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"GPU: {info['cuda_device_name']}")
        print(f"VRAM: {info['cuda_memory_gb']:.1f} GB")
        print(f"Compute: {info['cuda_compute_capability']}")
    
    flash_info = info['flash_attention']
    print(f"Flash Attention: {flash_info['available']}")
    
    if flash_info['recommendation']:
        print(f"💡 {flash_info['recommendation']}")


# Configuração de logging
def setup_logging(level: str = "INFO"):
    """
    Configura logging para o pacote.
    
    Args:
        level: Nível de logging ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar logger do pacote
    logger = logging.getLogger("libervia_core")
    logger.setLevel(getattr(logging, level.upper()))


if __name__ == "__main__":
    # Quando executado diretamente, mostrar informações
    print_system_info()
    print()
    print_model_info()