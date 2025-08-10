#!/usr/bin/env python3
"""
Exemplo de uso do libervia-ia-Core v1
Demonstra as principais funcionalidades do modelo
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Adicionar o package ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

from libervia_core import (
    LiberviaConfig,
    LiberviaForCausalLM,
    create_model,
    create_config,
    print_model_info,
    check_system_compatibility,
)


def example_1_basic_usage():
    """Exemplo 1: Uso básico do modelo."""
    print("📝 Exemplo 1: Uso Básico")
    print("-" * 30)
    
    # Criar modelo rapidamente
    model = create_model("2b")
    
    print(f"Modelo criado: {model.config.model_name}")
    print(f"Parâmetros: {model.num_parameters():,}")
    print(f"Memória: {model.get_memory_footprint():.1f} MB")
    
    # Teste básico
    input_ids = torch.randint(0, 1000, (1, 5))
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits']
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("✅ Uso básico funcionando!")


def example_2_custom_config():
    """Exemplo 2: Configuração customizada."""
    print("\n📝 Exemplo 2: Configuração Customizada")
    print("-" * 40)
    
    # Configuração personalizada
    config = create_config("2b", 
        max_sequence_length=8192,
        torch_dtype="bfloat16",
        use_flash_attention=True,
        dropout=0.1
    )
    
    print(f"Configuração customizada:")
    print(f"  Sequência máxima: {config.max_sequence_length}")
    print(f"  Tipo de dados: {config.torch_dtype}")
    print(f"  Flash Attention: {config.use_flash_attention}")
    print(f"  Dropout: {config.dropout}")
    
    # Criar modelo com config customizada
    model = LiberviaForCausalLM(config)
    print(f"✅ Modelo criado com config customizada!")


def example_3_inference_optimization():
    """Exemplo 3: Otimizações para inferência."""
    print("\n📝 Exemplo 3: Otimizações para Inferência")
    print("-" * 45)
    
    # Configuração otimizada para inferência
    config = create_config("2b",
        dropout=0.0,              # Desligar dropout
        use_kv_cache=True,        # Usar KV cache
        torch_dtype="float16"     # Menor precisão
    )
    
    model = LiberviaForCausalLM(config)
    model.eval()  # Modo inferência
    
    print("🎯 Comparando performance com e sem KV cache...")
    
    # Prompt inicial
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Sem cache (regenerar tudo)
    start_time = time.time()
    with torch.no_grad():
        for i in range(5):
            # Simular regeneração completa
            current_ids = prompt_ids[:, :i+5]
            outputs = model(current_ids)
    no_cache_time = time.time() - start_time
    
    # Com cache
    start_time = time.time()
    past_key_values = None
    with torch.no_grad():
        for i in range(5):
            if i == 0:
                # Primeira passada
                current_ids = prompt_ids
                outputs = model(current_ids, use_cache=True)
                past_key_values = outputs['past_key_values']
            else:
                # Próximos tokens
                next_token = torch.randint(0, config.vocab_size, (1, 1))
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs['past_key_values']
    
    cache_time = time.time() - start_time
    
    print(f"Sem cache: {no_cache_time:.4f}s")
    print(f"Com cache: {cache_time:.4f}s") 
    print(f"Speedup: {no_cache_time/cache_time:.1f}x")
    print("✅ KV cache acelera inferência!")


def example_4_text_generation():
    """Exemplo 4: Geração de texto simples."""
    print("\n📝 Exemplo 4: Geração de Texto")
    print("-" * 35)
    
    # Modelo pequeno para geração rápida
    config = create_config("2b")
    config = config.update(d_model=256, n_layers=4, vocab_size=1000)
    
    model = LiberviaForCausalLM(config)
    model.eval()
    
    print("🤖 Gerando sequência de tokens...")
    
    # Parâmetros de geração
    max_new_tokens = 15
    temperature = 0.8
    top_p = 0.9
    
    # Prompt inicial (tokens aleatórios como exemplo)
    generated_ids = torch.randint(0, config.vocab_size, (1, 5))
    
    print(f"Prompt inicial: {generated_ids.tolist()[0]}")
    
    # Geração token por token
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs['logits']
            
            # Logits do último token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remover tokens com probabilidade cumulativa > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Adicionar à sequência
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
    
    print(f"Sequência final: {generated_ids.tolist()[0]}")
    print(f"Tokens gerados: {max_new_tokens}")
    print("✅ Geração de texto funcionando!")


def example_5_model_comparison():
    """Exemplo 5: Comparação entre tamanhos."""
    print("\n📝 Exemplo 5: Comparação de Modelos")
    print("-" * 40)
    
    model_sizes = ["2b", "4b", "7b"]
    
    print(f"{'Modelo':<10} {'Parâmetros':<15} {'Memória (MB)':<12} {'Tempo (ms)':<12}")
    print("-" * 55)
    
    for size in model_sizes:
        try:
            # Configs reduzidas para teste rápido
            config = create_config(size)
            config = config.update(
                d_model=min(config.d_model, 512),
                n_layers=min(config.n_layers, 2)
            )
            
            model = LiberviaForCausalLM(config)
            
            # Teste de performance
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            params = model.num_parameters()
            memory = model.get_memory_footprint()
            
            print(f"{size.upper():<10} {params:<15,} {memory:<12.1f} {inference_time:<12.1f}")
            
        except Exception as e:
            print(f"{size.upper():<10} {'ERRO':<15} {str(e)[:20]:<12}")
    
    print("✅ Comparação de modelos concluída!")


def example_6_save_and_load():
    """Exemplo 6: Salvar e carregar configuração."""
    print("\n📝 Exemplo 6: Salvar e Carregar")
    print("-" * 35)
    
    # Criar configuração
    config = create_config("2b", max_sequence_length=4096)
    
    # Salvar configuração
    save_path = Path("./temp_config")
    save_path.mkdir(exist_ok=True)
    
    config.save_pretrained(save_path)
    print(f"✅ Configuração salva em: {save_path}")
    
    # Carregar configuração
    loaded_config = LiberviaConfig.from_pretrained(save_path)
    print(f"✅ Configuração carregada: {loaded_config.model_name}")
    
    # Verificar se são iguais
    assert config.d_model == loaded_config.d_model
    assert config.n_layers == loaded_config.n_layers
    print("✅ Configurações são idênticas!")
    
    # Limpeza
    import shutil
    shutil.rmtree(save_path)
    print("✅ Arquivos temporários removidos")


def example_7_cuda_usage():
    """Exemplo 7: Uso com CUDA (se disponível)."""
    print("\n📝 Exemplo 7: Uso com CUDA")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA não disponível, pulando exemplo")
        return
    
    print(f"🎮 GPU disponível: {torch.cuda.get_device_name()}")
    
    # Modelo para CUDA
    config = create_config("2b")
    config = config.update(d_model=256, n_layers=2, torch_dtype="float16")
    
    model = LiberviaForCausalLM(config)
    model = model.cuda().half()  # Mover para GPU e converter para FP16
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).cuda()
    
    # Benchmark CPU vs GPU
    model_cpu = LiberviaForCausalLM(config).eval()
    input_cpu = torch.randint(0, config.vocab_size, (1, 10))
    
    # CPU
    start_time = time.time()
    with torch.no_grad():
        outputs_cpu = model_cpu(input_cpu)
    cpu_time = time.time() - start_time
    
    # GPU
    start_time = time.time()
    with torch.no_grad():
        outputs_gpu = model(input_ids)
    gpu_time = time.time() - start_time
    
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print("✅ CUDA funcionando!")


def main():
    """Função principal - executa todos os exemplos."""
    print("🌟 libervia-ia-Core v1 - Exemplos de Uso")
    print("=" * 50)
    
    # Informações do sistema
    system_info = check_system_compatibility()
    print(f"Sistema: Python {system_info['python_version']}, PyTorch {system_info['torch_version']}")
    print(f"CUDA: {system_info['cuda_available']}")
    
    # Lista de exemplos
    examples = [
        example_1_basic_usage,
        example_2_custom_config,
        example_3_inference_optimization,
        example_4_text_generation,
        example_5_model_comparison,
        example_6_save_and_load,
        example_7_cuda_usage,
    ]
    
    # Executar exemplos
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Erro no exemplo {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Todos os exemplos executados!")
    print("\n💡 Dicas para uso:")
    print("  - Use create_model('2b') para início rápido")
    print("  - Configure dropout=0.0 para inferência")
    print("  - Use torch_dtype='float16' para economizar memória")
    print("  - Ative use_kv_cache=True para geração sequencial")
    print("  - Monitore uso de memória com model.get_memory_footprint()")


if __name__ == "__main__":
    main()