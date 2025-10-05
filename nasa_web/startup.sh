#!/bin/bash
# Activar entorno virtual si es necesario
# source ../.venv/bin/activate

# Activar entorno virtual
source .venv/bin/activate
# Instalar dependencias
pip install -r nasa_web/requirements.txt
# Ejecutar la aplicación con Gunicorn
exec gunicorn -b 0.0.0.0:5000 nasa_web.app:app
