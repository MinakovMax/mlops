#!/bin/bash

# Путь к файлу .env
ENV_FILE="./infra/.env"

# Проверка наличия gh CLI
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) не установлен."
    exit 1
fi

# Проверка авторизации
if ! gh auth status &>/dev/null; then
    echo "❌ Не авторизован в gh. Выполни 'gh auth login'"
    exit 1
fi

# Получение текущего репозитория
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

echo "📦 Загрузка переменных из $ENV_FILE в репозиторий $REPO"

# Загрузка каждой переменной из .env
while IFS='=' read -r key value; do
    # Пропускаем пустые строки и комментарии
    if [[ -z "$key" || "$key" == \#* ]]; then
        continue
    fi
    echo "🔐 Устанавливается секрет: $key"
    gh secret set "$key" --repo "$REPO" --body "$value"
done < "$ENV_FILE"

echo "✅ Все переменные окружения загружены как secrets GitHub."
