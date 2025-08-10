# 🌟 libervia-ia-Core v1

> Modelo de linguagem otimizado para soberania digital e uso descentralizado

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📖 Visão Geral

O **libervia-ia-Core** é um modelo de linguagem Transformer otimizado para:

- 🚀 **Performance**: Rápido e leve para hardware popular (8-16GB RAM)
- 🌐 **Descentralização**: Distribuição via IPFS/P2P
- 🇧🇷 **Linguagem Popular**: Treinado em português brasileiro, comunidades e DAOs
- 🔧 **Edge-First**: Execução eficiente em PCs ARM/x86
- 🛡️ **Soberania Digital**: Independente de big techs

### 🎯 Características Técnicas

| Modelo | Parâmetros | RAM Mín. | GPU Mín. | Uso Principal |
|--------|------------|----------|----------|---------------|
| Core-2B | 2.1B | 8GB | 6GB | Edge, mobile |
| Core-4B | 4.3B | 12GB | 8GB | Desktop, servidor |
| Core-7B | 7.2B | 16GB | 12GB | Workstation |

## 🚀 Quick Start

### Pré-requisitos
```bash
python --version  # 3.9+
```

### Instalação
```bash
# Setup completo
make setup
source venv/bin/activate

# Verificar instalação  
make status
make test
```

### Uso Rápido
```python
from libervia_core import LiberviaModel, LiberviaConfig

# Carregar modelo
config = LiberviaConfig.from_pretrained("libervia-2b")
model = LiberviaModel.from_pretrained("libervia-2b")

# Inferência
response = model.generate("Como funciona uma DAO?", max_length=256)
print(response)
```

### CLI
```bash
# Treinar modelo
make train MODEL=2b

# Servidor HTTP
make serve PORT=8000

# Export para GGUF
make export FORMAT=gguf MODEL=checkpoints/libervia-2b
```

## 🛠️ Desenvolvimento

### Comandos Make
```bash
make help           # Lista todos comandos
make setup          # Setup inicial completo
make dev            # Setup desenvolvimento
make test           # Executar testes
make lint           # Linting e formatação
make train          # Treinar modelo
make infer          # Servidor de inferência
make build          # Build para produção
make release        # Release para IPFS
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/amazing-feature`)
3. Commit suas mudanças (`git commit -m 'Add amazing feature'`)
4. Push para a branch (`git push origin feature/amazing-feature`)
5. Abra um Pull Request

Consulte [CONTRIBUTING.md](docs/CONTRIBUTING.md) para mais detalhes.

## 📄 Licença

Distribuído sob a licença Apache 2.0. Veja `LICENSE` para mais informações.
