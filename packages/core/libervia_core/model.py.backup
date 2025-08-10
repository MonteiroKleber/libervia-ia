"""
libervia-ia-Core v1 - Main Model
Implementação completa do modelo Transformer decoder-only
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint

from .config import LiberviaConfig
from .attention import create_attention_layer
from .components import RMSNorm, LiberviaFFN, KVCache


class LiberviaEmbedding(nn.Module):
    """
    Embedding layer otimizado para libervia-ia.
    
    Combina token embeddings com scaling apropriado.
    Suporte a tie_word_embeddings para compartilhar pesos com output layer.
    
    Args:
        config: Configuração do modelo
    """
    
    def __init__(self, config: LiberviaConfig) -> None:
        super().__init__()
        
        self.vocab_size = config.vocab_size
        self.hidden_size = config.d_model
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.d_model,
            padding_idx=getattr(config, 'pad_token_id', None)
        )
        
        # Scaling para estabilidade (usado em alguns modelos)
        self.embed_scale = config.d_model ** 0.5 if getattr(config, 'scale_embedding', False) else 1.0
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do embedding.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        embeddings = self.embed_tokens(input_ids)
        
        if self.embed_scale != 1.0:
            embeddings = embeddings * self.embed_scale
            
        return embeddings


class LiberviaDecoderLayer(nn.Module):
    """
    Uma camada do decoder Transformer.
    
    Implementa a arquitetura pré-norm:
    1. RMSNorm -> Self-Attention -> Residual
    2. RMSNorm -> FFN -> Residual
    
    Args:
        config: Configuração do modelo
        layer_idx: Índice da camada (para debugging)
    """
    
    def __init__(self, config: LiberviaConfig, layer_idx: int) -> None:
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.d_model
        
        # Self-attention
        self.self_attn = create_attention_layer(config, layer_idx)
        
        # Feed-forward network
        self.mlp = LiberviaFFN(config)
        
        # Layer normalization (pre-norm style)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # Dropout residual (se configurado)
        self.residual_dropout = nn.Dropout(config.residual_dropout) if config.residual_dropout > 0.0 else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[torch.Tensor]]:
        """
        Forward pass da camada decoder.
        
        Args:
            hidden_states: Estados ocultos [batch_size, seq_len, hidden_size]
            attention_mask: Máscara de atenção
            position_ids: IDs de posição
            past_key_value: Cache KV anterior
            output_attentions: Se deve retornar pesos de atenção
            use_cache: Se deve usar cache
            cache_position: Posição no cache
            
        Returns:
            Tuple (hidden_states, present_key_value, attn_weights)
        """
        residual = hidden_states
        
        # 1. Pre-norm + Self-attention + Residual
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states, present_key_value, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        # Residual dropout
        if self.residual_dropout is not None:
            hidden_states = self.residual_dropout(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # 2. Pre-norm + FFN + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual dropout
        if self.residual_dropout is not None:
            hidden_states = self.residual_dropout(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class LiberviaModel(nn.Module):
    """
    Modelo Transformer decoder-only principal.
    
    Este é o core do libervia-ia-Core, implementando um decoder Transformer
    otimizado para geração de linguagem com características modernas:
    - RMSNorm ao invés de LayerNorm
    - RoPE para codificação posicional
    - SwiGLU FFN
    - Flash Attention v2
    - Sliding Window attention
    - KV-Cache para inferência eficiente
    
    Args:
        config: Configuração do modelo
    """
    
    def __init__(self, config: LiberviaConfig) -> None:
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.d_model
        self.num_layers = config.n_layers
        
        # Embeddings
        self.embed_tokens = LiberviaEmbedding(config)
        
        # Layers do transformer
        self.layers = nn.ModuleList([
            LiberviaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.n_layers)
        ])
        
        # Norma final
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Inicialização dos pesos
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        """Retorna layer de input embeddings."""
        return self.embed_tokens.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Define layer de input embeddings."""
        self.embed_tokens.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Dict[str, Any]]:
        """
        Forward pass do modelo.
        
        Args:
            input_ids: IDs dos tokens [batch_size, seq_len]
            attention_mask: Máscara de atenção [batch_size, seq_len]
            position_ids: IDs de posição [batch_size, seq_len]
            past_key_values: Cache KV de chamadas anteriores
            inputs_embeds: Embeddings diretos (alternativa a input_ids)
            use_cache: Se deve retornar cache para próxima iteração
            output_attentions: Se deve retornar pesos de atenção
            output_hidden_states: Se deve retornar estados ocultos
            return_dict: Se deve retornar dict ou tuple
            cache_position: Posição atual no cache
            
        Returns:
            ModelOutput com last_hidden_state, past_key_values, etc.
        """
        # Configurações padrão
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Obter embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Você deve especificar input_ids ou inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Gerenciar cache
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Estados ocultos iniciais
        hidden_states = inputs_embeds
        
        # Armazenar outputs opcionais
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Passar pelas camadas
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Cache KV desta camada
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Usar gradient checkpointing
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    use_reentrant=False,
                )
            else:
                # Forward normal
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # Norma final
        hidden_states = self.norm(hidden_states)
        
        # Adicionar estados ocultos finais
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Preparar output
        if not return_dict:
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if use_cache:
                outputs += (next_decoder_cache,)
            return outputs
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache if use_cache else None,
            'hidden_states': all_hidden_states if output_hidden_states else None,
            'attentions': all_self_attns if output_attentions else None,
        }
    
    def post_init(self) -> None:
        """
        Inicialização dos pesos após construção do modelo.
        
        Aplica inicialização seguindo melhores práticas:
        - Xavier/Glorot para linear layers
        - Normal para embeddings
        - Zero para bias (quando presente)
        """
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Inicializa pesos de um módulo específico.
        
        Args:
            module: Módulo para inicializar
        """
        if isinstance(module, nn.Linear):
            # Xavier initialization para linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization para embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            # Inicializar pesos da norma como 1
            torch.nn.init.ones_(module.weight)
    
    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Conta número de parâmetros do modelo.
        
        Args:
            only_trainable: Se deve contar apenas parâmetros treináveis
            
        Returns:
            Número total de parâmetros
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self) -> float:
        """
        Calcula footprint de memória em MB.
        
        Returns:
            Memória utilizada em MB
        """
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_buffers = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        mem = mem_params + mem_buffers
        return mem / (1024 * 1024)  # Convert to MB


class LiberviaForCausalLM(nn.Module):
    """
    Modelo libervia-ia para Language Modeling (geração de texto).
    
    Adiciona a camada de output (lm_head) sobre o modelo base
    para predição de próximo token.
    
    Args:
        config: Configuração do modelo
    """
    
    def __init__(self, config: LiberviaConfig) -> None:
        super().__init__()
        
        self.config = config
        self.model = LiberviaModel(config)
        
        # Cabeça de language modeling
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights se configurado
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.embed_tokens.weight
        
        # Inicialização
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        """Retorna layer de input embeddings."""
        return self.model.embed_tokens.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Define layer de input embeddings."""
        self.model.embed_tokens.embed_tokens = value
    
    def get_output_embeddings(self) -> nn.Module:
        """Retorna layer de output embeddings."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        """Define layer de output embeddings."""
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Dict[str, Any]]:
        """
        Forward pass para language modeling.
        
        Args:
            input_ids: IDs dos tokens
            attention_mask: Máscara de atenção
            position_ids: IDs de posição
            past_key_values: Cache KV
            inputs_embeds: Embeddings diretos
            labels: Labels para cálculo de loss
            use_cache: Se deve usar cache
            output_attentions: Se deve retornar atenções
            output_hidden_states: Se deve retornar estados ocultos
            return_dict: Se deve retornar dict
            cache_position: Posição no cache
            
        Returns:
            CausalLMOutput com loss, logits, etc.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward do modelo base
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        if return_dict:
            hidden_states = outputs['last_hidden_state']
        else:
            hidden_states = outputs[0]
        
        # Projeção para vocabulário
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Garantir float32 para estabilidade
        
        # Calcular loss se labels fornecidos
        loss = None
        if labels is not None:
            # Shift para predição de próximo token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calcular cross-entropy loss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        result = {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions'),
        }
        
        return result
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[KVCache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepara inputs para geração de texto.
        
        Args:
            input_ids: IDs dos tokens
            past_key_values: Cache KV
            attention_mask: Máscara de atenção
            inputs_embeds: Embeddings diretos
            cache_position: Posição no cache
            **kwargs: Argumentos adicionais
            
        Returns:
            Dict com inputs preparados
        """
        # Se usando cache, usar apenas último token
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        
        position_ids = None
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def post_init(self) -> None:
        """Inicialização pós-construção."""
        self.model.post_init()
        
        # Inicializar lm_head se não tied
        if not self.config.tie_word_embeddings:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def num_parameters(self, only_trainable: bool = False) -> int:
        """Conta parâmetros do modelo completo."""
        return self.model.num_parameters(only_trainable)
    
    def get_memory_footprint(self) -> float:
        """Calcula footprint de memória."""
        return self.model.get_memory_footprint()


# Alias para compatibilidade
LiberviaLMHeadModel = LiberviaForCausalLM


if __name__ == "__main__":
    # Teste básico do modelo
    from .config import LiberviaConfig
    
    print("Testando libervia-ia-Core modelo...")
    
    config = LiberviaConfig.from_model_size("2b")
    model = LiberviaForCausalLM(config)
    
    print(f"Modelo criado: {config.model_name}")
    print(f"Parâmetros: {model.num_parameters():,}")
    print(f"Memória: {model.get_memory_footprint():.1f} MB")
    
    # Teste forward
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        print(f"Output logits shape: {logits.shape}")
        print("✅ Teste básico passou!")