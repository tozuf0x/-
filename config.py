from os import environ
from dotenv import load_dotenv

# загружаем переменные окружения

load_dotenv()

DATABASE_URL = environ.get("DATABASE_URL")
