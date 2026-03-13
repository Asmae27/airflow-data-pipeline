FROM apache/airflow:2.9.2-python3.11

# Passer en root pour installer les dépendances système
USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git \
 && apt-get clean

# Revenir à l'utilisateur airflow
USER airflow

RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    prophet==1.1.5 \
    scikit-learn \
    astral
