import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"

class ProcessLog:
    def __init__(self):
        self.entries = []
        self.start_time = datetime.now()
        self.current_stage = None
    
    def log(self, agent_name: str, stage: str, data: Any, status: AgentStatus, details: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "stage": stage,
            "status": status.value,
            "data": data,
            "details": details,
            "elapsed_time": (datetime.now() - self.start_time).total_seconds()
        }
        self.entries.append(entry)
        self.current_stage = stage
        
        log_msg = f"[{agent_name}] {stage}: {status.value}"
        if details:
            log_msg += f" - {details}"
        logging.info(log_msg)
    
    def get_stage_data(self, stage: str) -> Optional[Dict]:
        for entry in reversed(self.entries):
            if entry["stage"] == stage and entry["status"] in ["completed", "verified"]:
                return entry["data"]
        return None
    
    def get_agent_data(self, agent_name: str) -> Optional[Dict]:
        for entry in reversed(self.entries):
            if entry["agent"] == agent_name and entry["status"] in ["completed", "verified"]:
                return entry["data"]
        return None
    
    def save_to_file(self, filepath: str = "agentic/process.log"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)

class BaseAgent(ABC):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        self.model_id = model_id
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.token_usage = []
    
    def get_existing_cache_by_filename(self, file_path: str) -> Optional[str]:
        """Check if a cache already exists for this file based on filename pattern"""
        try:
            file_name = Path(file_path).stem
            expected_display_name = f"cache_{file_name}"
            
            # List all cached contents to find existing cache
            for cache in self.client.caches.list():
                if cache.display_name == expected_display_name:
                    self.logger.info(f"Found existing cache for {file_path}: {cache.name}")
                    return cache.name
            return None
        except Exception as e:
            self.logger.warning(f"Error checking for existing cache: {e}")
            return None
    
    def upload_and_cache_file(self, file_path: str, reuse_cache: bool = True) -> Dict[str, Any]:
        """Universal method to upload file using GenAI File API and create cache"""
        try:
            # Check for existing cache first if reuse is enabled
            if reuse_cache:
                existing_cache = self.get_existing_cache_by_filename(file_path)
                if existing_cache:
                    # Get cache details
                    cache = self.client.caches.get(name=existing_cache)
                    self.logger.info(f"Reusing existing cache for {file_path}: {existing_cache}")
                    return {
                        "file_object": None,  # Not needed when reusing cache
                        "cache_name": existing_cache,
                        "file_id": None,  # Not available when reusing cache
                        "reused": True
                    }
            
            # Upload file using the client
            uploaded_file = self.client.files.upload(file=file_path)
            self.logger.info(f"Uploaded file {file_path} with file ID: {uploaded_file.name}")
            
            # Determine mime type based on file extension
            file_extension = Path(file_path).suffix.lower()
            mime_type_map = {
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.csv': 'text/csv',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif'
            }
            mime_type = mime_type_map.get(file_extension, 'application/octet-stream')
            
            # Create cache for the uploaded file using correct syntax with file URI
            file_name = Path(file_path).stem
            cache = self.client.caches.create(
                model=self.model_id,
                config=types.CreateCachedContentConfig(
                    contents=[
                        types.Content(
                            role='user',
                            parts=[
                                types.Part.from_uri(
                                    file_uri=uploaded_file.uri,
                                    mime_type=mime_type
                                )
                            ]
                        )
                    ],
                    display_name=f"cache_{file_name}",
                    ttl='3600s'
                )
            )
            cache_name = cache.name
            self.logger.info(f"Created cache for {file_path} with cache name: {cache_name}")
            
            return {
                "file_object": uploaded_file,
                "cache_name": cache_name,
                "file_id": uploaded_file.name,
                "reused": False
            }
        except Exception as e:
            self.logger.error(f"Failed to upload and cache file {file_path}: {e}")
            raise
    
    def _generate_response(self, prompt: List[Any], temperature: float = 0.3, max_tokens: int = 800) -> tuple[str, Dict[str, Any]]:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,   
        )
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config
        )

        usage_metadata = response.usage_metadata
        token_count = {
            "prompt": usage_metadata.prompt_token_count,
            "candidates": usage_metadata.candidates_token_count,
            "total": usage_metadata.total_token_count
        }
        self.token_usage.append(token_count)
        self.logger.info(f"Token usage: {token_count}")
        
        return response.text, token_count

    def get_total_token_usage(self) -> Dict[str, int]:
        total_usage = {
            "prompt": sum(usage.get("prompt", 0) or 0 for usage in self.token_usage),
            "candidates": sum(usage.get("candidates", 0) or 0 for usage in self.token_usage),
            "total": sum(usage.get("total", 0) or 0 for usage in self.token_usage)
        }
        return total_usage

    @abstractmethod
    def execute(self, process_log: ProcessLog, **kwargs) -> Dict[str, Any]:
        pass