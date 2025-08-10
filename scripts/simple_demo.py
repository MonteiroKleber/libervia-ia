#!/usr/bin/env python3
"""
Demo simples de geração de texto com libervia-ia
Mostra a arquitetura funcionando mesmo sem treinamento
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

from libervia_core import LiberviaConfig, LiberviaForCausalLM

def simple_generation_demo():
    """Demo simples de geração."""
    print("🤖 Demo Simples - libervia-ia Geração")
    print("=" * 40)
    
    # Criar modelo pequeno
    config = LiberviaConfig(
        model_name="demo-tiny",
        d_model=64,
        n_layers=1, 
        n_heads=2,
        vocab_size=100,
        max_sequence_length=20,
        torch_dtype="float32",
        use_flash_attention=False,
    )
    
    model = LiberviaForCausalLM(config)
    model.eval()
    
    print(f"✅ Modelo: {model.num_parameters():,} parâmetros")
    
    # Simular input (tokens aleatórios)
    prompt_tokens = [5, 23, 47, 12]  # "Olá como você está" (simulado)
    print(f"📝 Prompt simulado: {prompt_tokens}")
    
    # Converter para tensor
    input_ids = torch.tensor([prompt_tokens])
    
    print("🔄 Gerando resposta...")
    
    # Gerar sequência
    generated = input_ids.clone()
    
    with torch.no_grad():
        for i in range(10):  # 10 novos tokens
            # Forward pass
            outputs = model(generated)
            logits = outputs['logits']
            
            # Próximo token (greedy)
            next_token = torch.argmax(logits[0, -1, :])
            
            # Adicionar à sequência
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            print(f"  Token {i+1}: {next_token.item()}")
    
    print(f"🎯 Sequência final: {generated[0].tolist()}")
    print(f"📊 Gerou {len(generated[0]) - len(prompt_tokens)} novos tokens")
    
    print("\n💡 Para um chat real, você precisa:")
    print("  1. Tokenizer (texto ↔ números)")
    print("  2. Modelo treinado (pesos úteis)")
    print("  3. Prompt engineering (formato de conversa)")

if __name__ == "__main__":
    simple_generation_demo()