# Verificar se todos os arquivos estão criados
ls -la packages/core/libervia_core/
ls -la packages/core/tests/
ls -la scripts/

# Deve mostrar:
# packages/core/libervia_core/:
#   __init__.py  config.py  components.py  attention.py  model.py
# packages/core/tests/:
#   test_model.py
# scripts/:
#   test_model.py  example_usage.py