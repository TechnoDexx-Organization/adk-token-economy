Понял. Куда именно добавить — зависит от того, к чему подключаешься. Вот два варианта.

### 🖥️ Если это LM Studio (локально)

URL: `http://127.0.0.1:1234/v1`  
API Key: **не нужен** (LM Studio не требует ключа для локального доступа).

В `agent.py`:

```python
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

os.environ["LM_STUDIO_API_BASE"] = "http://127.0.0.1:1234/v1"

root_agent = Agent(
    model=LiteLlm(model="lm_studio/имя_модели"),
    ...
)
```

### ☁️ Если это внешний API (DeepSeek, Selectel, Yandex)

URL и ключ задаются **переменными окружения** (не в коде):

```bash
export DEEPSEEK_API_KEY="твой_ключ"
# или для OpenAI-совместимых:
export OPENAI_API_KEY="твой_ключ"
export OPENAI_API_BASE="https://api.deepseek.com/v1"
```

А в `agent.py`:

```python
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(model="deepseek/deepseek-chat"),
    ...
)
```

LiteLLM сам подхватит ключ из переменной окружения.

### 📌 Где именно вписать

- **URL** — либо в `os.environ` (если LM Studio), либо в `OPENAI_API_BASE` (если внешний API).
- **API Key** — только через `export` в терминале, **не в коде**. Чтобы не попал в git.

Если скажешь, к какому именно провайдеру подключаешься — уточню точные имена переменных. 😎

## Переменные можно в .env поместить - он для этого там и есть.

Да, всё верно. `.env` для этого и создан. В ADK он подхватывается автоматически через `python-dotenv`, который уже установлен (я видел его в списке пакетов).

### 📄 Как это работает

В папке проекта создай файл `.env` и положи туда переменные:

```bash
# .env
DEEPSEEK_API_KEY=твой_ключ
OPENAI_API_BASE=https://api.deepseek.com/v1
```

Или для LM Studio (ключ не нужен, только URL):

```bash
# .env
LM_STUDIO_API_BASE=http://127.0.0.1:1234/v1
```

Главное — **загрузить** `.env` до создания агента. В ADK это делается одной строкой в начале `agent.py`:

```python
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(model="deepseek/deepseek-chat"),
    ...
)
```

`load_dotenv()` читает `.env` и добавляет переменные в окружение. После этого LiteLLM сам их подхватит.

### ⚠️ Важно

Добавь `.env` в `.gitignore`. Чтобы ключи случайно не улетели в репозиторий. 😎
