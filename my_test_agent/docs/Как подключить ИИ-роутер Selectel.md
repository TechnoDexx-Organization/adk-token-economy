root_agent = Agent(
    model=LiteLlm(
        # 1. ID модели из файла Selectel
        model="openai/deepseek/deepseek-v4.1-flash", 
        # 2. Эндпоинт Selectel (OpenAI-совместимый)
        api_base="https://api.selectel.ru/aig/v1",
        # 3. Ключ Selectel (добавишь в .env)
        api_key=os.getenv("SELECTEL_API_KEY"),
    ),
    # ... остальное без изменений ...
)
