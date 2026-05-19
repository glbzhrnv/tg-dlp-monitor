from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    api_id: int
    api_hash: str
    debug: bool = True

    class Config:
        env_file = "../../.env"  # относительный путь от app/ к корню

settings = Settings()