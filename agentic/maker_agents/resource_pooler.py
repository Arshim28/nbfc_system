import os
import json
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field
from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import logging
import pandas as pd

class DocumentMetadata(BaseModel):
    """Pydantic model for structured document metadata output"""
    model_config = {"extra": "forbid"}  # This prevents additionalProperties in JSON schema
    
    name: str = Field(description="The filename of the document")
    summary: str = Field(description="A concise one-paragraph summary of the document content")
    analyst_info: str = Field(description="Information about what an investment analyst can expect from this document")

class ResourcePoolerAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.logger = logging.getLogger("ResourcePoolerAgent")
        self.log_dir = Path("agentic/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "resource_pooler.log"

    def _extract_sheets_to_csv(self, excel_file_path: str, output_dir: str = "extracted_sheets") -> Dict[str, str]:
        self.logger.info(f"Extracting sheets from Excel file: {excel_file_path}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        excel_file = pd.ExcelFile(excel_file_path, engine="openpyxl")
        excel_file_name = Path(excel_file_path).stem
        csv_files = {}

        for sheet_name in excel_file.sheet_names:
            self.logger.debug(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")
            if df.shape[0] >= 1 and df.shape[1] >= 1:
                safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in [" ", "_"]).rstrip()
                csv_filename = f"{excel_file_name}_{safe_sheet_name}.csv"
                csv_path = Path(output_dir) / csv_filename
                df.to_csv(csv_path, index=False)
                csv_files[sheet_name] = str(csv_path)
                self.logger.info(f"Extracted sheet '{sheet_name}' to {csv_path}")
            else:
                self.logger.warning(f"Sheet '{sheet_name}' skipped due to insufficient data: shape={df.shape}")
        excel_file.close()
        self.logger.info(f"Completed extraction for {excel_file_path}, {len(csv_files)} sheets extracted.")
        return csv_files

    def _generate_metadata(self, file_path: str, file_info: Dict[str, Any]) -> dict:
        file_name = Path(file_path).name
        cache_name = file_info["cache_name"]
        reused_cache = file_info.get("reused", False)
        
        # If cache was reused, we can generate metadata using the cached content
        if reused_cache:
            prompt = f"""You are an assistant for investment analysts. Based on the cached document content, provide a structured analysis.

File name: {file_name}

Please analyze the document and provide structured information about it."""
        else:
            file_object = file_info["file_object"]
            prompt = [
                f"""You are an assistant for investment analysts. Read the following document and provide a structured analysis.

File name: {file_name}

Please analyze the document and provide structured information about it.""",
                file_object
            ]
        
        try:
            # Use structured output with dictionary schema instead of Pydantic model
            # Google GenAI doesn't fully support Pydantic models in response_schema
            response_schema = {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The filename of the document"
                    },
                    "summary": {
                        "type": "string", 
                        "description": "A concise one-paragraph summary of the document content"
                    },
                    "analyst_info": {
                        "type": "string",
                        "description": "Information about what an investment analyst can expect from this document"
                    }
                },
                "required": ["name", "summary", "analyst_info"]
            }
            
            config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=600,
                response_mime_type='application/json',
                response_schema=response_schema
            )
            
            if reused_cache:
                # Use cached content for generation
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=600,
                        response_mime_type='application/json',
                        response_schema=response_schema,
                        cached_content=cache_name
                    )
                )
            else:
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
            
            # Parse the structured response
            try:
                metadata_dict = json.loads(response.text)
                # Validate with Pydantic model (only core fields)
                metadata_model = DocumentMetadata(**metadata_dict)
                
                # Start with validated core fields
                final_metadata = metadata_model.model_dump()
                
                # Add additional fields not in the Pydantic model
                final_metadata.update({
                    "file_path": str(file_path),
                    "cache_name": cache_name,
                    "file_id": file_info.get("file_id", ""),
                    "token_usage": token_count,
                    "reused_cache": reused_cache
                })
                
                return final_metadata
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Structured output parsing failed for {file_path}: {e}")
                # Fallback to manual parsing if structured output fails
                return self._fallback_metadata_generation(file_path, file_info, response.text)
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata for {file_path}: {e}")
            return {
                "name": file_name, 
                "summary": "Could not summarize due to processing error.", 
                "analyst_info": "File upload successful but metadata generation failed.", 
                "file_path": str(file_path),
                "cache_name": cache_name,
                "file_id": file_info.get("file_id", ""),
                "token_usage": {},
                "reused_cache": reused_cache,
                "error": str(e)
            }

    def _fallback_metadata_generation(self, file_path: str, file_info: Dict[str, Any], response_text: str) -> dict:
        """Fallback method for when structured output fails"""
        file_name = Path(file_path).name
        cache_name = file_info["cache_name"]
        reused_cache = file_info.get("reused", False)
        
        # Try to extract JSON from response text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                metadata = json.loads(json_match.group(0))
                metadata.update({
                    "file_path": str(file_path),
                    "cache_name": cache_name,
                    "file_id": file_info.get("file_id", ""),
                    "reused_cache": reused_cache
                })
                return metadata
            except json.JSONDecodeError:
                pass
        
        # Complete fallback
        return {
            "name": file_name,
            "summary": "Document analysis completed but response formatting failed. Manual review recommended.",
            "analyst_info": "File processed successfully but metadata extraction encountered formatting issues.",
            "file_path": str(file_path),
            "cache_name": cache_name,
            "file_id": file_info.get("file_id", ""),
            "reused_cache": reused_cache,
            "fallback_used": True
        }

    def execute(self, process_log: ProcessLog, data_directory: str) -> None:
        self.logger.info(f"Starting execute() for data_directory: {data_directory}")
        process_log.log(self.__class__.__name__, "document_harvest", "Starting metadata generation", AgentStatus.RUNNING)
        data_path = Path(data_directory)
        processed_files = 0
        cache_names = []
        reused_caches = 0
        new_caches = 0
        
        with open(self.log_file, "w") as log_f:
            for file_path in data_path.rglob("*"):
                if file_path.is_file():
                    self.logger.info(f"Processing file: {file_path}")
                    try:
                        if file_path.suffix.lower() in [".xlsx", ".xls"]:
                            # Extract sheets to CSV and process each
                            csv_files = self._extract_sheets_to_csv(str(file_path))
                            for sheet_name, csv_path in csv_files.items():
                                # Upload and cache CSV file using universal method with cache reuse
                                file_info = self.upload_and_cache_file(csv_path, reuse_cache=True)
                                cache_names.append(file_info["cache_name"])
                                
                                if file_info.get("reused", False):
                                    reused_caches += 1
                                    self.logger.info(f"Reused existing cache for {csv_path}")
                                else:
                                    new_caches += 1
                                    self.logger.info(f"Created new cache for {csv_path}")
                                
                                # Generate metadata using the file object or cached content
                                metadata = self._generate_metadata(csv_path, file_info)
                                log_f.write(json.dumps(metadata) + "\n")
                                processed_files += 1
                        else:
                            # Upload and cache file directly using universal method with cache reuse
                            file_info = self.upload_and_cache_file(str(file_path), reuse_cache=True)
                            cache_names.append(file_info["cache_name"])
                            
                            if file_info.get("reused", False):
                                reused_caches += 1
                                self.logger.info(f"Reused existing cache for {file_path}")
                            else:
                                new_caches += 1
                                self.logger.info(f"Created new cache for {file_path}")
                            
                            # Generate metadata using the file object or cached content
                            metadata = self._generate_metadata(str(file_path), file_info)
                            log_f.write(json.dumps(metadata) + "\n")
                            processed_files += 1
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
        
        # Log comprehensive summary
        self.logger.info(f"Cache Statistics: {new_caches} new caches created, {reused_caches} existing caches reused")
        self.logger.info(f"Created {len(cache_names)} total caches: {cache_names}")
        self.logger.info(f"Processed {processed_files} files. Metadata written to {self.log_file}")
        
        process_log.log(
            self.__class__.__name__, 
            "document_harvest", 
            {
                "processed_files": processed_files,
                "cache_names": cache_names,
                "new_caches": new_caches,
                "reused_caches": reused_caches,
                "total_token_usage": self.get_total_token_usage()
            }, 
            AgentStatus.COMPLETED
        )

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/maker_agents/resource_pooler.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    from agentic.base.base_agent import ProcessLog
    process_log = ProcessLog()
    agent = ResourcePoolerAgent()
    agent.execute(process_log, data_dir)
    print("\n" + "="*50)
    print("RESOURCE POOLER AGENT COMPLETED")
    print("="*50)
    print(f"Metadata written to agentic/logs/resource_pooler.log")
    print(f"Total token usage: {agent.get_total_token_usage()}")
    print("="*50)