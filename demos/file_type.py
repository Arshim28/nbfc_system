import os
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

file_path = Path("__file__").parent.parent / "data" / "DTDJAn25check.pdf"
file_name = "DTDJAn25check.pdf"

prompt = f"""
Classify this document for a gold loan NBFC investment analysis based on its content:

Filename: {file_name}

Based on the content, determine:
1. Document type from these categories:
   - annual_report (audited financial statements)
   - quarterly_results (quarterly financial pack)
   - rating_rationale (credit rating report)
   - rbi_inspection (regulatory inspection report)
   - auditor_lfar (long form audit report)
   - debenture_docs (NCD documentation)
   - gold_tonnage_data (branch-wise gold stock data)
   - alm_statements (asset liability management)
   - mca_events (corporate registry events)
   - crilc_history (credit bureau data)
   - other

2. Fiscal period (if applicable): FY2021, FY2022, etc.
3. Indicative usefulness (1-5 scale for investment analysis)

Respond in JSON format:
{{"doc_type": "category", "fiscal_period": "FYXXXX", "indicative_usefulness": X, "confidence": X}}
"""

upload_file = client.files.upload(file=file_path)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite-preview-06-17",
    contents=[upload_file, prompt],
)

print(response.text)