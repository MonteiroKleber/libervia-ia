#!/usr/bin/env python3
"""
Script automático PORTÁVEL para aplicar as correções do libervia-ia-Core v1
PODE SER EXECUTADO DE QUALQUER PASTA - detecta automaticamente a raiz do projeto

Uso: python auto_fix_portable.py
"""

import os
import shutil
import sys
from pathlib import Path

def find_project_root():
    """Encontra a raiz do projeto libervia-ia automaticamente."""
    current = Path.cwd()
    
    # Verificar se já estamos na raiz
    if (current / "packages" / "core" / "libervia_core").exists():
        return current
    
    # Procurar para cima na árvore de diretórios
    for parent in [current] + list(current.parents):
        if (parent / "packages" / "core" / "libervia_core").exists():
            print(f"📍 Projeto encontrado em: {parent}")
            return parent
    
    # Procurar para baixo na árvore (caso esteja em um diretório pai)
    for item in current.rglob("*"):
        if item.name == "libervia_core" and item.is_dir():
            # Verificar se é a estrutura correta
            if (item.parent.name == "core" and 
                item.parent.parent.name == "packages"):
                root = item.parent.parent.parent
                print(f"📍 Projeto encontrado em: {root}")
                return root
    
    return None

def backup_files(project_root):
    """Cria backup dos arquivos originais."""
    print("📦 Criando backup dos arquivos originais...")
    
    config_file = project_root / "packages" / "core" / "libervia_core" / "config.py"
    model_file = project_root / "packages" / "core" / "libervia_core" / "model.py"
    
    if config_file.exists():
        shutil.copy2(config_file, str(config_file) + ".backup")
        print(f"   ✅ Backup: {config_file}.backup")
    
    if model_file.exists():
        shutil.copy2(model_file, str(model_file) + ".backup")
        print(f"   ✅ Backup: {model_file}.backup")

def apply_config_fix(project_root):
    """Aplica correções no config.py."""
    print("\n🔧 Aplicando correções em config.py...")
    
    config_content = '''"""
libervia-ia-Core v1 - Model Configuration  
Configuração compatível com HuggingFace transformers
CORREÇÃO: Herdar de PretrainedConfig para compatibilidade HF
"""

from __future__ import annotations

import json
import math
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import torch
from transformers import PretrainedConfig


class LiberviaConfig(PretrainedConfig):
    """
    Configuração do modelo libervia-ia-Core compatível com HuggingFace.
    
    Esta classe define todos os hiperparâmetros arquiteturais e de treinamento
    do modelo Transformer decoder-only, seguindo as melhores práticas modernas.
    """
    
    model_type = "libervia"
    
    def __init__(
        self,
        # Identificação
        model_name: str = "libervia-core",
        version: str = "1.0.0",
        
        # Arquitetura Principal
        d_model: int = 2048,
        n_layers: int = 22,
        n_heads: int = 16,
        head_dim: Optional[int] = None,
        ffn_mult: float = 2.67,
        vocab_size: int = 32000,
        
        # Contexto e Posição
        max_sequence_length: int = 4096,
        sliding_window: Optional[int] = None,
        rope_base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        
        # Normalização
        norm_eps: float = 1e-6,
        norm_type: str = "rmsnorm",
        
        # Otimizações
        use_flash_attention: bool = True,
        use_kv_cache: bool = True,
        use_sliding_window: bool = True,
        tie_word_embeddings: bool = True,
        scale_attn_weights: bool = True,
        
        # Treinamento
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        
        # Quantização
        quantization_config: Optional[Dict[str, Any]] = None,
        
        # Tipos de dados
        torch_dtype: str = "float16",
        
        # HuggingFace específicos
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        
        **kwargs
    ):
        # Configurar defaults para HuggingFace
        kwargs.setdefault("use_return_dict", True)
        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("use_cache", True)
        
        # Preservar total_params se fornecido em kwargs (não quebrar)
        self.total_params = kwargs.pop("total_params", None)
        self.memory_footprint_mb = kwargs.pop("memory_footprint_mb", None)
        
        # Chamar init da superclasse
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        # Nossos atributos específicos
        self.model_name = model_name
        self.version = version
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.ffn_mult = ffn_mult
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.sliding_window = sliding_window
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.norm_eps = norm_eps
        self.norm_type = norm_type
        self.use_flash_attention = use_flash_attention
        self.use_kv_cache = use_kv_cache
        self.use_sliding_window = use_sliding_window
        self.scale_attn_weights = scale_attn_weights
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.quantization_config = quantization_config
        self.torch_dtype = torch_dtype
        
        # Post-init validation e cálculos
        self._post_init()
    
    def _post_init(self) -> None:
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
        
        # Calcular parâmetros totais se não fornecido
        if self.total_params is None:
            self.total_params = self.estimate_parameters()
        if self.memory_footprint_mb is None:
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
        hidden_dim = int(self.d_model * self.ffn_mult)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Múltiplo de 64
        
        per_layer_params = (
            # Attention: q, k, v, o projections
            4 * (self.d_model * self.d_model) +
            # FFN: gate, up, down projections  
            3 * (self.d_model * hidden_dim) +
            # RMSNorm weights (2 per layer)
            2 * self.d_model
        )
        
        transformer_params = self.n_layers * per_layer_params
        final_norm_params = self.d_model
        
        # Projeção de saída
        if self.tie_word_embeddings:
            output_params = 0
        else:
            output_params = self.vocab_size * self.d_model
        
        total = token_embed_params + transformer_params + final_norm_params + output_params
        return total
    
    def estimate_memory_footprint(self) -> float:
        """Estima footprint de memória em MB."""
        if self.total_params is None:
            params = self.estimate_parameters()
        else:
            params = self.total_params
        
        bytes_per_param = {
            "float32": 4,
            "float16": 2, 
            "bfloat16": 2
        }
        
        bytes_per_param_value = bytes_per_param[self.torch_dtype]
        total_bytes = params * bytes_per_param_value
        return total_bytes / (1024 * 1024)
    
    @classmethod
    def from_model_size(cls, size: str) -> 'LiberviaConfig':
        """Factory method para criar configurações baseadas no tamanho."""
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
            raise ValueError(f"Tamanho não suportado: {size}. Use '2b', '4b' ou '7b'")
    
    def get_torch_dtype(self) -> torch.dtype:
        """Converte string de dtype para torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        return dtype_map[self.torch_dtype]
    
    def update(self, **kwargs) -> 'LiberviaConfig':
        """Cria nova configuração com parâmetros atualizados."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


# Manter compatibilidade com código existente
LIBERVIA_2B_CONFIG = LiberviaConfig.from_model_size("2b")
LIBERVIA_4B_CONFIG = LiberviaConfig.from_model_size("4b") 
LIBERVIA_7B_CONFIG = LiberviaConfig.from_model_size("7b")

def get_config_for_size(size: str) -> LiberviaConfig:
    """Função helper para obter configuração por tamanho."""
    return LiberviaConfig.from_model_size(size)
'''
    
    config_file = project_root / "packages" / "core" / "libervia_core" / "config.py"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("   ✅ config.py atualizado com herança PretrainedConfig")

def apply_model_fix(project_root):
    """Aplica correções no model.py."""
    print("\n🔧 Aplicando correções em model.py...")
    
    # Ler arquivo original
    model_file = project_root / "packages" / "core" / "libervia_core" / "model.py"
    if not model_file.exists():
        print(f"   ❌ Arquivo não encontrado: {model_file}")
        return
    
    with open(model_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aplicar correções linha por linha
    
    # 1. Adicionar imports HF
    hf_imports = '''
# Imports HuggingFace para compatibilidade
try:
    from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
except ImportError:
    # Fallback se transformers não disponível
    CausalLMOutputWithPast = None
    BaseModelOutputWithPast = None
'''
    
    # Inserir após a linha dos imports torch
    if "import torch.utils.checkpoint" in content:
        content = content.replace(
            "import torch.utils.checkpoint",
            "import torch.utils.checkpoint" + hf_imports
        )
    
    # 2. Corrigir getattr fallbacks
    replacements = [
        ("self.config.output_attentions", "getattr(self.config, 'output_attentions', False)"),
        ("self.config.output_hidden_states", "getattr(self.config, 'output_hidden_states', False)"), 
        ("self.config.use_cache", "getattr(self.config, 'use_cache', True)"),
        ("self.config.use_return_dict", "getattr(self.config, 'use_return_dict', True)"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # 3. Corrigir returns para objetos HF
    # Encontrar e substituir returns dict por objetos HF
    old_return_model = """        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache if use_cache else None,
            'hidden_states': all_hidden_states if output_hidden_states else None,
            'attentions': all_self_attns if output_attentions else None,
        }"""
    
    new_return_model = """        # Retornar BaseModelOutputWithPast se disponível, senão dict
        if BaseModelOutputWithPast is not None:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache if use_cache else None,
                hidden_states=all_hidden_states if output_hidden_states else None,
                attentions=all_self_attns if output_attentions else None,
            )
        else:
            return {
                'last_hidden_state': hidden_states,
                'past_key_values': next_decoder_cache if use_cache else None,
                'hidden_states': all_hidden_states if output_hidden_states else None,
                'attentions': all_self_attns if output_attentions else None,
            }"""
    
    content = content.replace(old_return_model, new_return_model)
    
    # 4. Corrigir acesso a outputs
    content = content.replace(
        "hidden_states = outputs['last_hidden_state']",
        """if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs['last_hidden_state']"""
    )
    
    # 5. Corrigir return final do CausalLM
    old_return_causal = """        result = {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions'),
        }
        
        return result"""
    
    new_return_causal = """        # Retornar CausalLMOutputWithPast se disponível, senão dict
        if CausalLMOutputWithPast is not None:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=getattr(outputs, 'past_key_values', None),
                hidden_states=getattr(outputs, 'hidden_states', None),
                attentions=getattr(outputs, 'attentions', None),
            )
        else:
            result = {
                'loss': loss,
                'logits': logits,
                'past_key_values': outputs.get('past_key_values') if hasattr(outputs, 'get') else getattr(outputs, 'past_key_values', None),
                'hidden_states': outputs.get('hidden_states') if hasattr(outputs, 'get') else getattr(outputs, 'hidden_states', None),
                'attentions': outputs.get('attentions') if hasattr(outputs, 'get') else getattr(outputs, 'attentions', None),
            }
            return result"""
    
    content = content.replace(old_return_causal, new_return_causal)
    
    # Salvar arquivo corrigido
    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ model.py atualizado com objetos HF compatíveis")

def test_fixes(project_root):
    """Testa se as correções funcionaram."""
    print("\n🧪 Testando correções...")
    
    try:
        # Adicionar o path do projeto
        sys.path.insert(0, str(project_root / "packages" / "core"))
        
        from libervia_core import LiberviaConfig, create_model
        import torch
        
        # Teste 1: Config com total_params
        config = LiberviaConfig.from_model_size("2b", total_params=123456)
        assert hasattr(config, 'use_return_dict'), "Deve ter use_return_dict"
        print("   ✅ Config aceita kwargs e tem atributos HF")
        
        # Teste 2: Modelo funciona
        model = create_model("2b")
        input_ids = torch.randint(0, 1000, (1, 5))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert 'logits' in outputs or hasattr(outputs, 'logits'), "Deve ter logits"
        print("   ✅ Forward funciona sem erros")
        
        # Teste 3: Cache funciona
        with torch.no_grad():
            outputs_cache = model(input_ids, use_cache=True)
        
        past_kv = outputs_cache.get('past_key_values') if hasattr(outputs_cache, 'get') else getattr(outputs_cache, 'past_key_values', None)
        assert past_kv is not None, "Cache deve funcionar"
        print("   ✅ KV Cache funciona")
        
        print("\n🎉 TODAS AS CORREÇÕES APLICADAS COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal."""
    print("🚀 Auto-Fix PORTÁVEL libervia-ia-Core v1 - Correção HuggingFace")
    print("🌍 Executável de qualquer pasta - detecta automaticamente o projeto")
    print("=" * 70)
    
    # Encontrar raiz do projeto
    project_root = find_project_root()
    
    if project_root is None:
        print("❌ Projeto libervia-ia não encontrado!")
        print("   Certifique-se de estar em um diretório que contém o projeto")
        print("   Estrutura esperada: .../libervia-ia/packages/core/libervia_core/")
        sys.exit(1)
    
    print(f"📂 Trabalhando em: {project_root}")
    
    # Mudar para a raiz do projeto temporariamente
    original_cwd = Path.cwd()
    os.chdir(project_root)
    
    try:
        # Passo 1: Backup
        backup_files(project_root)
        
        # Passo 2: Aplicar correções
        apply_config_fix(project_root)
        apply_model_fix(project_root)
        
        # Passo 3: Testar
        if test_fixes(project_root):
            print("\n" + "=" * 70)
            print("✅ CORREÇÕES APLICADAS COM SUCESSO!")
            print(f"\n📍 Projeto corrigido em: {project_root}")
            print("\nPróximos passos:")
            print(f"1. cd {project_root}")
            print("2. python scripts/test_model.py")
            print("3. git add . && git commit -m 'fix(core): HF compatibility'")
            print("\n🎯 Objetivo: Taxa de sucesso 100% nos testes críticos!")
        else:
            print("\n❌ Alguma correção falhou. Verifique os logs acima.")
            print("💡 Para reverter: use os arquivos .backup criados")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Erro durante aplicação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Voltar ao diretório original
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()