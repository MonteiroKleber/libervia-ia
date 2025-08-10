"""
Testes unitários para libervia-ia-Core
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libervia_core import (
    LiberviaConfig,
    LiberviaModel,
    LiberviaForCausalLM,
    create_model,
    create_config,
    check_flash_attention_availability,
    RMSNorm,
    RotaryPositionalEmbedding,
    KVCache,
)


class TestLiberviaConfig:
    """Testes para LiberviaConfig."""
    
    def test_config_creation(self):
        """Testa criação básica de configuração."""
        config = LiberviaConfig()
        assert config.d_model == 2048
        assert config.n_layers == 22
        assert config.n_heads == 16
        assert config.vocab_size == 32000
    
    def test_config_from_size(self):
        """Testa factory method por tamanho."""
        for size in ["2b", "4b", "7b"]:
            config = LiberviaConfig.from_model_size(size)
            assert config.model_name.endswith(size)
            assert config.total_params is not None
            assert config.total_params > 0
    
    def test_config_validation(self):
        """Testa validação da configuração."""
        with pytest.raises(ValueError):
            LiberviaConfig(d_model=0)
        
        with pytest.raises(ValueError):
            LiberviaConfig(n_layers=0)
        
        with pytest.raises(ValueError):
            LiberviaConfig(dropout=1.5)
    
    def test_config_serialization(self):
        """Testa serialização da configuração."""
        config = LiberviaConfig.from_model_size("2b")
        
        # To dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["d_model"] == config.d_model
        
        # From dict
        config2 = LiberviaConfig.from_dict(config_dict)
        assert config2.d_model == config.d_model
        assert config2.model_name == config.model_name
    
    def test_config_update(self):
        """Testa atualização de configuração."""
        config = LiberviaConfig.from_model_size("2b")
        original_seq_len = config.max_sequence_length
        
        new_config = config.update(max_sequence_length=8192)
        assert new_config.max_sequence_length == 8192
        assert config.max_sequence_length == original_seq_len  # Original não mudou
    
    def test_parameter_estimation(self):
        """Testa estimativa de parâmetros."""
        config_2b = LiberviaConfig.from_model_size("2b")
        config_4b = LiberviaConfig.from_model_size("4b")
        
        assert config_2b.total_params < config_4b.total_params
        assert config_2b.memory_footprint_mb < config_4b.memory_footprint_mb


class TestComponents:
    """Testes para componentes base."""
    
    def test_rmsnorm(self):
        """Testa RMSNorm."""
        hidden_size = 512
        norm = RMSNorm(hidden_size)
        
        x = torch.randn(2, 10, hidden_size)
        output = norm(x)
        
        assert output.shape == x.shape
        assert torch.allclose(output.var(dim=-1, keepdim=True), torch.ones(2, 10, 1), atol=1e-3)
    
    def test_rotary_embedding(self):
        """Testa Rotary Position Embedding."""
        head_dim = 64
        max_pos = 2048
        rope = RotaryPositionalEmbedding(head_dim, max_pos)
        
        x = torch.randn(2, 8, 10, head_dim)  # batch, heads, seq, head_dim
        cos, sin = rope(x, seq_len=10)
        
        assert cos.shape == (10, head_dim)
        assert sin.shape == (10, head_dim)
    
    def test_kv_cache(self):
        """Testa KV Cache."""
        batch_size, seq_len = 2, 5
        n_heads, head_dim = 8, 64
        max_seq_len = 2048
        
        cache = KVCache(
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        
        # Teste update
        keys = torch.randn(batch_size, n_heads, seq_len, head_dim)
        values = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        cached_keys, cached_values = cache.update(keys, values, 0)
        
        assert cached_keys.shape == (batch_size, n_heads, seq_len, head_dim)
        assert cached_values.shape == (batch_size, n_heads, seq_len, head_dim)
        assert torch.allclose(cached_keys, keys)
        assert torch.allclose(cached_values, values)


class TestModel:
    """Testes para modelo principal."""
    
    @pytest.fixture
    def config(self):
        """Configuração para testes."""
        return LiberviaConfig(
            d_model=256,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            max_sequence_length=128,
        )
    
    def test_model_creation(self, config):
        """Testa criação do modelo."""
        model = LiberviaModel(config)
        
        assert isinstance(model, nn.Module)
        assert model.config == config
        assert len(model.layers) == config.n_layers
    
    def test_model_forward(self, config):
        """Testa forward pass do modelo."""
        model = LiberviaModel(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
            
            if isinstance(outputs, dict):
                hidden_states = outputs['last_hidden_state']
            else:
                hidden_states = outputs[0]
            
            assert hidden_states.shape == (batch_size, seq_len, config.d_model)
    
    def test_causal_lm_model(self, config):
        """Testa modelo para language modeling."""
        model = LiberviaForCausalLM(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs[0]
            
            assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_model_with_labels(self, config):
        """Testa modelo com labels para treinamento."""
        model = LiberviaForCausalLM(config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        if isinstance(outputs, dict):
            loss = outputs['loss']
            logits = outputs['logits']
        else:
            loss = outputs[0]
            logits = outputs[1]
        
        assert loss is not None
        assert loss.item() > 0
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_model_with_cache(self, config):
        """Testa modelo com KV cache."""
        model = LiberviaForCausalLM(config)
        
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            # Primeira passada
            outputs1 = model(input_ids, use_cache=True)
            
            if isinstance(outputs1, dict):
                past_key_values = outputs1['past_key_values']
            else:
                past_key_values = outputs1[-1]
            
            assert past_key_values is not None
            assert len(past_key_values) == config.n_layers
            
            # Segunda passada com cache
            next_token = torch.randint(0, config.vocab_size, (batch_size, 1))
            outputs2 = model(
                next_token, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            if isinstance(outputs2, dict):
                logits2 = outputs2['logits']
            else:
                logits2 = outputs2[0]
            
            assert logits2.shape == (batch_size, 1, config.vocab_size)
    
    def test_parameter_count(self, config):
        """Testa contagem de parâmetros."""
        model = LiberviaForCausalLM(config)
        
        param_count = model.num_parameters()
        trainable_count = model.num_parameters(only_trainable=True)
        
        assert param_count > 0
        assert trainable_count == param_count  # Todos treináveis por padrão
        
        # Comparar com estimativa da config
        estimated = config.estimate_parameters()
        # Permitir pequena diferença devido a implementação
        assert abs(param_count - estimated) / estimated < 0.1
    
    def test_memory_footprint(self, config):
        """Testa cálculo de memory footprint."""
        model = LiberviaForCausalLM(config)
        
        footprint = model.get_memory_footprint()
        assert footprint > 0
        
        # Deve ser próximo da estimativa da config
        estimated = config.estimate_memory_footprint()
        assert abs(footprint - estimated) / estimated < 0.2


class TestFactoryFunctions:
    """Testes para funções factory."""
    
    def test_create_config(self):
        """Testa create_config."""
        config = create_config("2b")
        assert config.model_name == "libervia-core-2b"
        
        # Com kwargs
        config = create_config("2b", max_sequence_length=8192)
        assert config.max_sequence_length == 8192
    
    def test_create_model(self):
        """Testa create_model."""
        model = create_model("2b")
        assert isinstance(model, LiberviaForCausalLM)
        
        # Teste básico de forward
        input_ids = torch.randint(0, 1000, (1, 5))
        with torch.no_grad():
            outputs = model(input_ids)
            assert outputs is not None


class TestCompatibility:
    """Testes de compatibilidade do sistema."""
    
    def test_flash_attention_check(self):
        """Testa verificação de Flash Attention."""
        info = check_flash_attention_availability()
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'cuda_available' in info
    
    def test_torch_dtypes(self):
        """Testa compatibilidade com tipos de dados."""
        config = LiberviaConfig(torch_dtype="float16")
        assert config.get_torch_dtype() == torch.float16
        
        config = LiberviaConfig(torch_dtype="bfloat16")
        assert config.get_torch_dtype() == torch.bfloat16
        
        config = LiberviaConfig(torch_dtype="float32")
        assert config.get_torch_dtype() == torch.float32


class TestEdgeCases:
    """Testes para casos extremos."""
    
    def test_empty_sequence(self):
        """Testa com sequência vazia."""
        config = LiberviaConfig(d_model=128, n_layers=1, n_heads=2, vocab_size=100)
        model = LiberviaForCausalLM(config)
        
        # Sequência de 1 token
        input_ids = torch.randint(0, 100, (1, 1))
        
        with torch.no_grad():
            outputs = model(input_ids)
            assert outputs is not None
    
    def test_large_sequence(self):
        """Testa com sequência no limite."""
        config = LiberviaConfig(
            d_model=128, 
            n_layers=1, 
            n_heads=2, 
            vocab_size=100,
            max_sequence_length=64
        )
        model = LiberviaForCausalLM(config)
        
        # Sequência no limite
        input_ids = torch.randint(0, 100, (1, 64))
        
        with torch.no_grad():
            outputs = model(input_ids)
            assert outputs is not None
    
    def test_different_batch_sizes(self):
        """Testa com diferentes tamanhos de batch."""
        config = LiberviaConfig(d_model=128, n_layers=1, n_heads=2, vocab_size=100)
        model = LiberviaForCausalLM(config)
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 100, (batch_size, 10))
            
            with torch.no_grad():
                outputs = model(input_ids)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs[0]
                
                assert logits.shape[0] == batch_size


# Fixture para skip testes que requerem GPU
@pytest.fixture
def require_cuda():
    """Skip teste se CUDA não disponível."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA não disponível")


class TestCUDA:
    """Testes específicos para CUDA (se disponível)."""
    
    def test_model_cuda(self, require_cuda):
        """Testa modelo em CUDA."""
        config = LiberviaConfig(d_model=128, n_layers=1, n_heads=2, vocab_size=100)
        model = LiberviaForCausalLM(config).cuda()
        
        input_ids = torch.randint(0, 100, (1, 10)).cuda()
        
        with torch.no_grad():
            outputs = model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs[0]
            
            assert logits.is_cuda
            assert logits.shape == (1, 10, 100)


# Executar testes quando chamado diretamente
if __name__ == "__main__":
    # Testes básicos sem pytest
    print("🧪 Executando testes básicos...")
    
    # Teste config
    config = LiberviaConfig.from_model_size("2b")
    print(f"✅ Config criada: {config.model_name}")
    
    # Teste modelo
    model = create_model("2b")
    print(f"✅ Modelo criado: {model.num_parameters():,} parâmetros")
    
    # Teste forward
    input_ids = torch.randint(0, 1000, (1, 5))
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        print(f"✅ Forward pass: {logits.shape}")
    
    print("🎉 Todos os testes básicos passaram!")