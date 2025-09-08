import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    
    # Collection Configuration
    COLLECTION_NAME: str = "faq_collection"
    DIMENSION: int = 384  # sentence-transformers/all-MiniLM-L6-v2
    INDEX_TYPE: str = "IVF_FLAT"
    METRIC_TYPE: str = "COSINE"
    NLIST: int = 1024
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text Processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    
    # Search Configuration
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # File Watching
    FAQ_DATA_PATH: str = "./data/faq_data.json"
    WATCH_DIRECTORY: str = "./data"
    
    class Config:
        env_file = ".env"

settings = Settings()