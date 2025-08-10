#!/usr/bin/env python3
"""
Script de teste com configurações reduzidas para libervia-ia-Core v1
Versão otimizada para sistemas com RAM limitada
"""

import sys
import os
import time
import traceback
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

# Adicionar o package ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

try:
    from libervia_core import (
        LiberviaConfig,
        LiberviaForCausalLM,
        create_model,
        create_config,
        check_flash_attention_availability,
        check_system_compatibility,
        print_system_info,
        print_model_info,
    )
    print("✅ Imports libervia-core OK")
except ImportError as e:
    print(f"❌ Erro ao importar libervia-core: {e}")
    sys.exit(1)


def get_memory_usage():
    """Obtém uso atual de memória."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        return 0


def create_tiny_config(name="tiny"):
    """Cria configuração muito pequena para testes."""
    return LiberviaConfig(
        model_name=f"libervia-{name}",
        d_model=256,        # Muito pequeno
        n_layers=2,         # Apenas 2 camadas
        n_heads=4,          # 4 cabeças
        num_key_value_heads=4,
        ffn_mult=2.0,       # FFN menor
        vocab_size=1000,    # Vocabulário reduzido
        max_sequence_length=64,  # Sequência curta
        sliding_window=32,
        torch_dtype="float32",  # Usar float32 para compatibilidade CPU
        use_flash_attention=False,  # Desabilitar Flash Attention
        dropout=0.0,
        gradient_checkpointing=False,
    )


def test_basic_functionality():
    """Testa funcionalidades básicas com modelo pequeno."""
    print("\n🧪 Teste 1: Funcionalidades Básicas (Modelo Pequeno)")
    print("-" * 55)
    
    mem_before = get_memory_usage()
    
    try:
        # Teste de configuração pequena
        config = create_tiny_config("basic")
        print(f"✅ Config criada: {config.model_name}")
        print(f"   Dimensões: {config.d_model}d × {config.n_layers}L × {config.n_heads}H")
        print(f"   Parâmetros estimados: {config.total_params:,}")
        print(f"   Memória estimada: {config.memory_footprint_mb:.1f} MB")
        
        # Teste de modelo
        model = LiberviaForCausalLM(config)
        real_params = model.num_parameters()
        print(f"✅ Modelo criado: {type(model).__name__}")
        print(f"   Parâmetros reais: {real_params:,}")
        
        mem_after = get_memory_usage()
        print(f"   Memória usada: {mem_after - mem_before:.1f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste básico: {e}")
        traceback.print_exc()
        return False
    finally:
        # Limpeza
        gc.collect()


def test_forward_pass():
    """Testa forward pass com modelo pequeno."""
    print("\n🧪 Teste 2: Forward Pass (Modelo Pequeno)")
    print("-" * 45)
    
    try:
        config = create_tiny_config("forward")
        model = LiberviaForCausalLM(config)
        
        # Teste com tamanhos pequenos
        test_cases = [
            (1, 1),    # Mínimo
            (1, 5),    # Pequeno
            (2, 10),   # Batch pequeno
        ]
        
        for batch_size, seq_len in test_cases:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
                
            end_time = time.time()
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs[0]
            
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert logits.shape == expected_shape, f"Shape incorreta: {logits.shape} vs {expected_shape}"
            
            tokens_per_sec = (batch_size * seq_len) / (end_time - start_time)
            print(f"✅ Batch {batch_size}x{seq_len}: {logits.shape} - {tokens_per_sec:.1f} tok/s")
        
        return True
    except Exception as e:
        print(f"❌ Erro no forward pass: {e}")
        traceback.print_exc()
        return False
    finally:
        gc.collect()


def test_generation():
    """Testa geração simples."""
    print("\n🧪 Teste 3: Geração de Texto (Modelo Pequeno)")
    print("-" * 50)
    
    try:
        config = create_tiny_config("generation")
        model = LiberviaForCausalLM(config)
        model.eval()
        
        # Prompt muito pequeno
        prompt_ids = torch.randint(0, config.vocab_size, (1, 3))
        print(f"✅ Prompt inicial: {prompt_ids.tolist()}")
        
        # Geração de apenas 5 tokens
        generated_ids = prompt_ids.clone()
        max_new_tokens = 5
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = model(generated_ids)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs[0]
                
                # Próximo token (greedy)
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Adicionar à sequência
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        
        print(f"✅ Sequência gerada: {generated_ids.tolist()}")
        print(f"   Tokens gerados: {max_new_tokens}")
        
        return True
    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        traceback.print_exc()
        return False
    finally:
        gc.collect()


def test_kv_cache():
    """Testa KV cache."""
    print("\n🧪 Teste 4: KV Cache (Modelo Pequeno)")
    print("-" * 42)
    
    try:
        config = create_tiny_config("cache")
        model = LiberviaForCausalLM(config)
        model.eval()
        
        # Teste muito pequeno
        input_ids = torch.randint(0, config.vocab_size, (1, 3))
        
        start_time = time.time()
        with torch.no_grad():
            outputs1 = model(input_ids, use_cache=True)
        time_with_cache_init = time.time() - start_time
        
        if isinstance(outputs1, dict):
            past_key_values = outputs1['past_key_values']
        else:
            past_key_values = outputs1[-1]
        
        print(f"✅ Cache inicial criado: {len(past_key_values)} layers")
        
        # Segunda passada usando cache
        next_token = torch.randint(0, config.vocab_size, (1, 1))
        
        start_time = time.time()
        with torch.no_grad():
            outputs2 = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
        time_with_cache = time.time() - start_time
        
        print(f"✅ Inferência com cache: {time_with_cache:.4f}s")
        print(f"   Speedup potencial: {time_with_cache_init/time_with_cache:.1f}x")
        
        return True
    except Exception as e:
        print(f"❌ Erro no KV cache: {e}")
        traceback.print_exc()
        return False
    finally:
        gc.collect()


def test_config_operations():
    """Testa operações de configuração."""
    print("\n🧪 Teste 5: Operações de Configuração")
    print("-" * 42)
    
    try:
        # Teste de criação por tamanho (mas com override pequeno)
        config = create_config("2b")
        print(f"✅ Config 2B criada: {config.model_name}")
        
        # Override para tamanho pequeno
        small_config = config.update(
            d_model=128,
            n_layers=1,
            n_heads=2,
            vocab_size=500,
            max_sequence_length=32
        )
        print(f"✅ Config reduzida: {small_config.d_model}d × {small_config.n_layers}L")
        
        # Teste de serialização
        config_dict = small_config.to_dict()
        print("✅ Serialização: OK")
        
        # Teste de recriação
        new_config = LiberviaConfig.from_dict(config_dict)
        print("✅ Deserialização: OK")
        
        # Verificar igualdade
        assert new_config.d_model == small_config.d_model
        print("✅ Configurações são consistentes")
        
        return True
    except Exception as e:
        print(f"❌ Erro nas operações de config: {e}")
        traceback.print_exc()
        return False


def test_memory_monitoring():
    """Testa monitoramento de memória."""
    print("\n🧪 Teste 6: Monitoramento de Memória")
    print("-" * 42)
    
    try:
        mem_start = get_memory_usage()
        print(f"✅ Memória inicial: {mem_start:.1f} MB")
        
        # Criar modelo pequeno
        config = create_tiny_config("memory")
        mem_after_config = get_memory_usage()
        
        model = LiberviaForCausalLM(config)
        mem_after_model = get_memory_usage()
        
        # Teste forward
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        with torch.no_grad():
            outputs = model(input_ids)
        
        mem_after_forward = get_memory_usage()
        
        print(f"✅ Após config: +{mem_after_config - mem_start:.1f} MB")
        print(f"✅ Após modelo: +{mem_after_model - mem_after_config:.1f} MB")
        print(f"✅ Após forward: +{mem_after_forward - mem_after_model:.1f} MB")
        print(f"✅ Total usado: {mem_after_forward - mem_start:.1f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Erro no monitoramento: {e}")
        return False
    finally:
        gc.collect()


def main():
    """Função principal de teste com configurações reduzidas."""
    print("🚀 libervia-ia-Core v1 - Teste com Configurações Reduzidas")
    print("=" * 65)
    
    # Informações do sistema
    print_system_info()
    print()
    
    # Verificar memória disponível
    mem_total = get_memory_usage()
    print(f"💾 Memória atual do processo: {mem_total:.1f} MB")
    
    # Lista de testes otimizados
    tests = [
        ("Funcionalidades Básicas", test_basic_functionality),
        ("Forward Pass", test_forward_pass),
        ("Geração de Texto", test_generation),
        ("KV Cache", test_kv_cache),
        ("Configurações", test_config_operations),
        ("Monitoramento de Memória", test_memory_monitoring),
    ]
    
    # Executar testes
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        start_time = time.time()
        
        try:
            success = test_func()
            end_time = time.time()
            
            if success:
                print(f"✅ {test_name}: PASSOU ({end_time - start_time:.2f}s)")
                results.append((test_name, True, end_time - start_time))
            else:
                print(f"❌ {test_name}: FALHOU ({end_time - start_time:.2f}s)")
                results.append((test_name, False, end_time - start_time))
                
        except Exception as e:
            end_time = time.time()
            print(f"💥 {test_name}: ERRO - {e}")
            results.append((test_name, False, end_time - start_time))
        
        # Forçar garbage collection entre testes
        gc.collect()
        
        # Mostrar memória atual
        current_mem = get_memory_usage()
        print(f"   Memória atual: {current_mem:.1f} MB")
    
    total_time = time.time() - total_start
    
    # Relatório final
    print(f"\n🏁 RELATÓRIO FINAL")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Testes executados: {total}")
    print(f"Testes passou: {passed}")
    print(f"Testes falhou: {total - passed}")
    print(f"Taxa de sucesso: {passed/total*100:.1f}%")
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Memória final: {get_memory_usage():.1f} MB")
    
    print("\nDetalhes:")
    for test_name, success, test_time in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"  {test_name:<25} {status} ({test_time:.2f}s)")
    
    if passed == total:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print("libervia-ia-Core está funcionando corretamente!")
        print("\n💡 Para testar modelos maiores:")
        print("   - Aumente a RAM disponível")
        print("   - Use GPU com CUDA")
        print("   - Configure quantização")
        return 0
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam.")
        print("Verifique os erros acima.")
        return 1


if __name__ == "__main__":
    sys.exit(main())