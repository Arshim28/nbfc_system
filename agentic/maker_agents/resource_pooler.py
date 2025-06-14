import os
import time
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
    model_id = config["resource_pooler"]["model"]

class ResourcePooler:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _extract_sheets_to_csv(self, excel_file_path: str, output_dir: str = "output") -> Dict[str, str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        excel_file = pd.ExcelFile(excel_file_path, engine="openpyxl")
        excel_file_name = Path(excel_file_path).stem
        csv_files = {}

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")
            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in [" ", "_"]).rstrip()
            csv_filename = f"{excel_file_name}_{safe_sheet_name}.csv"
            csv_path = Path(output_dir) / csv_filename
            df.to_csv(csv_path, index=False)
            csv_files[sheet_name] = str(csv_path)

        excel_file.close()
        return csv_files

    def glance_csv(self, csv_file_path: str) -> str:
        """TBD"""
        pass

    def glance_pdf(self, pdf_file_path: str) -> str:
        """TBD"""
        pass
    
    def generate_overview(self) -> str:
        """TBD"""
        pass


    