from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

@dataclass
class DatasetConfig:
    name:str
    sample_size:int
    config_metadata:str
    config_content:str
    config_relationships:str

@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    separators: list[int]
    clean_works: int

@dataclass
class Config:
    dataset: DatasetConfig
    chunking: ChunkingConfig
 