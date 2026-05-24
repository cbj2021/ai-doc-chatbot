from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 4
    UPLOAD_DIR: str = "uploads"
    CHROMA_DIR: str = "chroma_db"

    class Config:
        env_file = ".env"

settings = Settings()
