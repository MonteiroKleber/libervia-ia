#!/usr/bin/env python3
"""
Demo de Chat para libervia-ia-Core v1
AVISO: Este é um demo simulado - o modelo NÃO foi treinado!
"""

import sys
import time
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F

# Adicionar o package ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

from libervia_core import LiberviaConfig, LiberviaForCausalLM


class DummyTokenizer:
    """
    Tokenizer simulado para demo.
    NOTA: Um tokenizer real usaria SentencePiece ou BPE.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        
        # Tokens especiais
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1, 
            "<bos>": 2,   # Begin of sequence
            "<eos>": 3,   # End of sequence
            "<user>": 4,  # Usuário
            "<assistant>": 5,  # Assistente
        }
        
        # Vocabulário simulado (palavras comuns)
        self.vocab = {
            # Tokens especiais
            **self.special_tokens,
            
            # Palavras básicas em português
            "olá": 10, "oi": 11, "bom": 12, "dia": 13, "noite": 14,
            "como": 15, "você": 16, "está": 17, "tudo": 18, "bem": 19,
            "obrigado": 20, "obrigada": 21, "por": 22, "favor": 23,
            "sim": 24, "não": 25, "talvez": 26, "claro": 27,
            "eu": 28, "sou": 29, "um": 30, "uma": 31, "modelo": 32,
            "de": 33, "linguagem": 34, "artificial": 35, "inteligente": 36,
            "help": 37, "ajuda": 38, "ajudar": 39, "resposta": 40,
            "pergunta": 41, "questão": 42, "dúvida": 43,
            "legal": 44, "interessante": 45, "incrível": 46,
            "python": 47, "código": 48, "programação": 49,
            "libervia": 50, "ia": 51, "core": 52,
        }
        
        # Criar vocabulário reverso
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Preencher resto com tokens dummy
        for i in range(len(self.vocab), vocab_size):
            self.id_to_token[i] = f"<unk_{i}>"
    
    def encode(self, text: str) -> List[int]:
        """Converte texto para tokens (muito simplificado)."""
        # Normalizar texto
        text = text.lower().strip()
        
        # Dividir em palavras
        words = text.split()
        
        # Converter para IDs
        token_ids = [self.special_tokens["<bos>"]]
        
        for word in words:
            # Remover pontuação básica
            word = word.strip(".,!?;:")
            
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                # Token desconhecido - usar hash simples
                token_id = hash(word) % (self.vocab_size - 100) + 100
                token_ids.append(token_id)
        
        token_ids.append(self.special_tokens["<eos>"])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Converte tokens para texto."""
        words = []
        
        for token_id in token_ids:
            if token_id in [0, 2, 3]:  # Skip special tokens
                continue
            
            if token_id in self.id_to_token:
                word = self.id_to_token[token_id]
                if not word.startswith("<"):
                    words.append(word)
            else:
                words.append(f"<{token_id}>")
        
        return " ".join(words)


class LiberviaChat:
    """
    Sistema de chat simulado para libervia-ia.
    AVISO: Modelo não treinado - respostas são aleatórias!
    """
    
    def __init__(self, model_size: str = "tiny"):
        print("🤖 Inicializando libervia-ia Chat...")
        
        # Configuração ultra pequena para demo
        self.config = LiberviaConfig(
            model_name=f"libervia-chat-{model_size}",
            d_model=128,      # Muito pequeno para demo
            n_layers=2,       # 2 camadas
            n_heads=4,        # 4 cabeças
            num_key_value_heads=4,
            ffn_mult=2.0,
            vocab_size=1000,  # Vocabulário pequeno
            max_sequence_length=128,
            torch_dtype="float32",
            use_flash_attention=False,
            dropout=0.0,
        )
        
        # Criar modelo
        self.model = LiberviaForCausalLM(self.config)
        self.model.eval()
        
        # Tokenizer simulado
        self.tokenizer = DummyTokenizer(self.config.vocab_size)
        
        # Parâmetros de geração
        self.generation_config = {
            "max_new_tokens": 20,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
        }
        
        print(f"✅ Modelo carregado: {self.model.num_parameters():,} parâmetros")
        print("⚠️  AVISO: Modelo NÃO foi treinado - respostas são simuladas!")
    
    def format_prompt(self, user_message: str, chat_history: List[Dict] = None) -> str:
        """Formata prompt para conversa."""
        prompt = ""
        
        # Adicionar histórico se fornecido
        if chat_history:
            for msg in chat_history[-3:]:  # Últimas 3 mensagens
                role = msg["role"]
                content = msg["content"]
                prompt += f"<{role}> {content} "
        
        # Adicionar mensagem atual
        prompt += f"<user> {user_message} <assistant>"
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Gera resposta do modelo."""
        try:
            # Tokenizar prompt
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids])
            
            print(f"🔤 Prompt tokenizado: {len(input_ids)} tokens")
            
            # Gerar resposta
            generated_ids = input_tensor.clone()
            
            with torch.no_grad():
                for i in range(self.generation_config["max_new_tokens"]):
                    # Forward pass
                    outputs = self.model(generated_ids)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
                    
                    # Logits do último token
                    next_token_logits = logits[0, -1, :] / self.generation_config["temperature"]
                    
                    # Sampling
                    if self.generation_config["do_sample"]:
                        # Top-p sampling
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy
                        next_token = torch.argmax(next_token_logits)
                    
                    # Verificar se é EOS
                    if next_token.item() == self.tokenizer.special_tokens["<eos>"]:
                        break
                    
                    # Adicionar à sequência
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Extrair apenas os novos tokens (resposta)
            response_ids = generated_ids[0, len(input_ids):].tolist()
            response_text = self.tokenizer.decode(response_ids)
            
            return response_text.strip()
            
        except Exception as e:
            print(f"❌ Erro na geração: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Resposta de fallback quando há erro."""
        responses = [
            "Interessante! Me conte mais sobre isso.",
            "Entendi. Como posso ajudar?",
            "Essa é uma boa pergunta.",
            "Vou pensar sobre isso.",
            "Posso ajudar com mais alguma coisa?",
            "Obrigado por compartilhar isso comigo.",
            "Como você está se sentindo hoje?",
            "Que legal! Continue.",
        ]
        return random.choice(responses)
    
    def chat(self):
        """Interface de chat interativo."""
        print("\n" + "="*50)
        print("🌟 libervia-ia Chat Demo")
        print("="*50)
        print("ℹ️  Digite 'sair' para terminar")
        print("ℹ️  Digite 'ajuda' para comandos")
        print("⚠️  LEMBRE-SE: Modelo não foi treinado!")
        print()
        
        chat_history = []
        
        while True:
            try:
                # Input do usuário
                user_input = input("👤 Você: ").strip()
                
                if not user_input:
                    continue
                
                # Comandos especiais
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    print("👋 Tchau! Obrigado por testar o libervia-ia!")
                    break
                
                if user_input.lower() in ['ajuda', 'help']:
                    print("📋 Comandos disponíveis:")
                    print("  - 'sair': Terminar chat")
                    print("  - 'limpar': Limpar histórico")
                    print("  - 'status': Mostrar status do modelo")
                    continue
                
                if user_input.lower() in ['limpar', 'clear']:
                    chat_history = []
                    print("🧹 Histórico limpo!")
                    continue
                
                if user_input.lower() in ['status']:
                    print(f"📊 Status do modelo:")
                    print(f"   Parâmetros: {self.model.num_parameters():,}")
                    print(f"   Configuração: {self.config.d_model}d × {self.config.n_layers}L")
                    print(f"   Histórico: {len(chat_history)} mensagens")
                    continue
                
                # Adicionar ao histórico
                chat_history.append({"role": "user", "content": user_input})
                
                # Formatar prompt
                prompt = self.format_prompt(user_input, chat_history)
                
                # Gerar resposta
                print("🤖 libervia-ia: ", end="", flush=True)
                
                start_time = time.time()
                response = self.generate_response(prompt)
                end_time = time.time()
                
                print(response)
                print(f"⏱️  ({end_time - start_time:.2f}s)")
                
                # Adicionar resposta ao histórico
                chat_history.append({"role": "assistant", "content": response})
                
                # Limitar histórico
                if len(chat_history) > 10:
                    chat_history = chat_history[-8:]  # Manter últimas 8 mensagens
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrompido. Tchau!")
                break
            except Exception as e:
                print(f"\n❌ Erro no chat: {e}")
                continue


def main():
    """Função principal do chat demo."""
    print("🚀 libervia-ia Chat Demo")
    print("=" * 30)
    
    try:
        # Criar sistema de chat
        chat_system = LiberviaChat("tiny")
        
        # Iniciar chat interativo
        chat_system.chat()
        
    except Exception as e:
        print(f"❌ Erro ao inicializar chat: {e}")
        print("\n💡 Certifique-se de que o libervia-core está funcionando:")
        print("   python scripts/quick_test.py")


if __name__ == "__main__":
    main()