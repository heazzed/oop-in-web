### Установка зависимостей:

```shell
pip install -r reqirements.txt
```

### Проверка типизации
```shell
mypy --strict --show-error-codes checkpoints/{number}/models.py
```

### Запуск тестов
```shell
pytest test.py
```

### Запуск веб-приложения
```shell
gunicorn app:app
```