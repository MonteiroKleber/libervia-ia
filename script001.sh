# Execute isso na pasta libervia-ia:
echo "Criando requirements.txt..."

cat > requirements.txt << 'EOF'
torch>=2.1.0
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
hydra-core>=1.3.0
pydantic>=2.4.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
tqdm>=4.66.0
rich>=13.6.0
EOF

cat > requirements-dev.txt << 'EOF'
pytest>=7.4.0
pytest-cov>=4.1.0
pre-commit>=3.5.0
mypy>=1.6.0
ruff>=0.1.0
black>=23.9.0
isort>=5.12.0
EOF

# Remover venv problemático
rm -rf venv/

# Setup com requirements.txt
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r requirements-dev.txt

echo "✅ Pronto! Execute: source venv/bin/activate"