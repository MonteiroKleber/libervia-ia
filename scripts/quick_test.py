#!/usr/bin/env python3
"""
Teste super rápido para verificar se libervia-ia está funcionando
"""

import sys
from pathlib import Path
import torch
import gc

# Adicionar o package ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

def quick_test():
    print("🚀 Teste Rápido libervia-ia-Core")
    print("=" * 40)
    
    try:
        # 1. Import
        from libervia_core import LiberviaConfig, LiberviaForCausalLM
        print("✅ 1. Import: OK")
        
        # 2. Config micro
        config = LiberviaConfig(
            model_name="libervia-micro",
            d_model=64,      # Super pequeno
            n_layers=1,      # 1 camada só
            n_heads=2,       # 2 cabeças
            num_key_value_heads=2,
            vocab_size=100,  # 100 tokens
            max_sequence_length=10,
            torch_dtype="float32",
            use_flash_attention=False,
            dropout=0.0
        )
        print(f"✅ 2. Config: OK ({config.total_params:,} params)")
        
        # 3. Modelo
        model = LiberviaForCausalLM(config)
        print(f"✅ 3. Modelo: OK ({model.num_parameters():,} params)")
        
        # 4. Forward pass
        input_ids = torch.randint(0, 100, (1, 5))
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        
        print(f"✅ 4. Forward: OK {input_ids.shape} -> {logits.shape}")
        
        # 5. Geração
        with torch.no_grad():
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            print(f"✅ 5. Geração: OK (próximo token: {next_token.item()})")
        
        print("\n🎉 SUCESSO! libervia-ia-Core está funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        gc.collect()

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n💡 Próximos passos:")
        print("   1. Execute: python scripts/memory_safe_test.py")
        print("   2. Para modelos maiores, adicione mais RAM ou use GPU")
        print("   3. Para instalar Flash Attention: pip install flash-attn --no-build-isolation")
    else:
        print("\n🔧 Para resolver problemas:")
        print("   1. Verifique se todas as dependências estão instaladas")
        print("   2. Execute: pip install torch numpy")
        print("   3. Reinicie o terminal e tente novamente")