#!/bin/bash

# Загружаем переменные из .env
if [ -f .env ]; then
  echo "Loading environment variables from .env"
  source .env
else
  echo ".env file not found! Aborting."
  exit 1
fi

# Инициализация Terraform (один раз)
terraform init

# Применение конфигурации с автоматическим подтверждением
terraform apply -auto-approve