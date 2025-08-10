#!/usr/bin/env python3
"""
Script de teste completo para libervia-ia-Core v1
Verifica se todas as funcionalidades estão funcionando corretamente
"""

import sys
import os
import time
import traceback
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


def test_basic_functionality():
    """Testa funcionalidades básicas."""
    print("\n🧪 Teste 1: Funcionalidades Básicas")
    print("-" * 40)
    
    try:
        # Teste de configuração
        config = LiberviaConfig.from_model_size("2b")
        print(f"✅ Config criada: {config.model_name}")
        print(f"   Parâmetros: {config.total_params:,}")
        print(f"   Memória: {config.memory_footprint_mb:.1f} MB")
        
        # Teste de modelo
        model = create_model("2b")
        print(f"✅ Modelo criado: {type(model).__name__}")
        print(f"   Parâmetros reais: {model.num_parameters():,}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste básico: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Testa forward pass do modelo."""
    print("\n🧪 Teste 2: Forward Pass")
    print("-" * 40)
    
    try:
        config = create_config("2b")
        model = LiberviaForCausalLM(config)
        
        # Teste com diferentes tamanhos
        test_cases = [
            (1, 1),    # Mínimo
            (1, 10),   # Pequeno
            (2, 20),   # Batch
            (1, 100),  # Sequência maior
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


def test_generation():
    """Testa geração de texto simples."""
    print("\n🧪 Teste 3: Geração de Texto")
    print("-" * 40)
    
    try:
        config = create_config("2b")
        model = LiberviaForCausalLM(config)
        model.eval()
        
        # Prompt inicial
        prompt_ids = torch.randint(0, config.vocab_size, (1, 5))
        print(f"✅ Prompt inicial: {prompt_ids.tolist()}")
        
        # Geração simples (sem sampling sofisticado)
        generated_ids = prompt_ids.clone()
        max_new_tokens = 10
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Forward pass
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


def test_kv_cache():
    """Testa KV cache para inferência eficiente."""
    print("\n🧪 Teste 4: KV Cache")
    print("-" * 40)
    
    try:
        config = create_config("2b")
        model = LiberviaForCausalLM(config)
        model.eval()
        
        # Primeira passada
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
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


def test_different_models():
    """Testa diferentes tamanhos de modelo."""
    print("\n🧪 Teste 5: Diferentes Tamanhos")
    print("-" * 40)
    
    model_sizes = ["2b", "4b", "7b"]
    
    for size in model_sizes:
        try:
            config = create_config(size)
            
            # Criar modelo pequeno para teste (reduzir para economizar memória)
            config = config.update(
                d_model=min(config.d_model, 512),
                n_layers=min(config.n_layers, 2),
                max_sequence_length=min(config.max_sequence_length, 64)
            )
            
            model = LiberviaForCausalLM(config)
            
            # Teste rápido
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            with torch.no_grad():
                outputs = model(input_ids)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs[0]
            
            print(f"✅ {size.upper()}: {model.num_parameters():,} params - {logits.shape}")
            
        except Exception as e:
            print(f"❌ Erro no modelo {size}: {e}")
            return False
    
    return True


def test_memory_usage():
    """Testa uso de memória."""
    print("\n🧪 Teste 6: Uso de Memória")
    print("-" * 40)
    
    try:
        import psutil
        process = psutil.Process()
        
        # Memória inicial
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Criar modelo
        config = create_config("2b")
        # Reduzir para teste
        config = config.update(d_model=256, n_layers=2)
        model = LiberviaForCausalLM(config)
        
        # Memória após criação
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        model_memory = mem_after - mem_before
        
        # Teste forward
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Memória após forward
        mem_forward = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"✅ Memória baseline: {mem_before:.1f} MB")
        print(f"   Memória modelo: +{model_memory:.1f} MB")
        print(f"   Memória forward: +{mem_forward - mem_after:.1f} MB")
        print(f"   Total: {mem_forward:.1f} MB")
        
        return True
    except ImportError:
        print("⚠️  psutil não disponível, pulando teste de memória")
        return True
    except Exception as e:
        print(f"❌ Erro no teste de memória: {e}")
        return False


def test_cuda_if_available():
    """Testa CUDA se disponível."""
    print("\n🧪 Teste 7: CUDA (se disponível)")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA não disponível, pulando teste")
        return True
    
    try:
        config = create_config("2b")
        # Modelo pequeno para CUDA
        config = config.update(d_model=256, n_layers=2)
        
        model = LiberviaForCausalLM(config).cuda()
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).cuda()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
        cuda_time = time.time() - start_time
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs[0]
        
        assert logits.is_cuda, "Output não está em CUDA"
        
        print(f"✅ CUDA OK: {cuda_time:.4f}s")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memória: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste CUDA: {e}")
        traceback.print_exc()
        return False


def benchmark_performance():
    """Benchmark básico de performance."""
    print("\n🧪 Benchmark: Performance")
    print("-" * 40)
    
    try:
        config = create_config("2b")
        # Modelo otimizado para benchmark
        config = config.update(d_model=512, n_layers=4)
        model = LiberviaForCausalLM(config)
        model.eval()
        
        # Diferentes cenários
        scenarios = [
            ("Batch=1, Seq=1", 1, 1),
            ("Batch=1, Seq=10", 1, 10),
            ("Batch=1, Seq=100", 1, 100),
            ("Batch=4, Seq=10", 4, 10),
        ]
        
        print(f"{'Cenário':<20} {'Tempo (ms)':<12} {'Tok/s':<10} {'Throughput':<15}")
        print("-" * 65)
        
        for name, batch_size, seq_len in scenarios:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # Warmup
            with torch.no_grad():
                model(input_ids)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(input_ids)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / avg_time
            
            print(f"{name:<20} {avg_time*1000:<12.1f} {tokens_per_sec:<10.1f} {total_tokens/avg_time:<15.1f}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no benchmark: {e}")
        return False


def main():
    """Função principal de teste."""
    print("🚀 libervia-ia-Core v1 - Teste Completo")
    print("=" * 50)
    
    # Informações do sistema
    print_system_info()
    print()
    
    # Lista de testes
    tests = [
        ("Funcionalidades Básicas", test_basic_functionality),
        ("Forward Pass", test_forward_pass),
        ("Geração de Texto", test_generation),
        ("KV Cache", test_kv_cache),
        ("Diferentes Modelos", test_different_models),
        ("Uso de Memória", test_memory_usage),
        ("CUDA", test_cuda_if_available),
    ]
    
    # Executar testes
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
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
    
    # Benchmark opcional
    print(f"\n{'='*20} Benchmark Opcional {'='*20}")
    try:
        benchmark_performance()
    except Exception as e:
        print(f"⚠️  Benchmark falhou: {e}")
    
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
    
    print("\nDetalhes:")
    for test_name, success, test_time in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"  {test_name:<25} {status} ({test_time:.2f}s)")
    
    if passed == total:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print("libervia-ia-Core está funcionando corretamente!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam.")
        print("Verifique os erros acima.")
        return 1


if __name__ == "__main__":
    sys.exit(main())