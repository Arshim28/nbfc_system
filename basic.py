import os
import time
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

import pandas as pd
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv

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
    
    def log(self, agent_name: str, stage: str, data: Any, status: AgentStatus):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "stage": stage,
            "status": status.value,
            "data": data,
            "elapsed_time": (datetime.now() - self.start_time).total_seconds()
        }
        self.entries.append(entry)
        logging.info(f"[{agent_name}] {stage}: {status.value}")
    
    def get_stage_data(self, stage: str) -> Optional[Dict]:
        for entry in reversed(self.entries):
            if entry["stage"] == stage and entry["status"] == "completed":
                return entry["data"]
        return None

class BaseAgent:
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        self.model_id = model_id
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, process_log: ProcessLog, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class ResourcePoolerAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.cached_files = {}
    
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
    
    def _get_or_create_cache(self, file_path: str, ttl: str = "3600s") -> Any:
        cache_name = f"cache_{Path(file_path).stem}"
        
        for cache in self.client.caches.list():
            if getattr(cache, "display_name", None) == cache_name:
                return cache
        
        uploaded_file = self.client.files.upload(file=file_path)
        cache = self.client.caches.create(
            model=self.model_id,
            config=types.CreateCachedContentConfig(
                contents=[uploaded_file],
                display_name=cache_name,
                ttl=ttl
            )
        )
        self.cached_files[file_path] = cache
        return cache
    
    def _analyze_csv_overview(self, csv_path: str) -> str:
        df = pd.read_csv(csv_path)
        
        prompt = f"""
        Analyze this CSV data and provide a concise overview for investment analysis purposes.
        
        Dataset Info:
        - Shape: {df.shape}
        - Columns: {df.columns.tolist()}
        - Data Types: {df.dtypes.to_dict()}
        
        Sample Data:
        {df.head(3).to_string()}
        
        Provide a 2-3 sentence overview focusing on:
        1. What type of financial/business data this appears to be
        2. Key metrics or categories present
        3. Relevance for investment analysis
        """
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=300
            )
        )
        return response.text
    
    def _analyze_pdf_overview(self, pdf_path: str) -> str:
        cache = self._get_or_create_cache(pdf_path)
        
        prompt = """
        Analyze this document and provide a concise overview for investment analysis purposes.
        
        Provide a 2-3 sentence overview focusing on:
        1. Document type (annual report, circular, presentation, etc.)
        2. Key financial or business information contained
        3. Relevance for investment decision-making
        4. Time period covered (if applicable)
        """
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                cached_content=cache.name,
                temperature=0.3,
                max_output_tokens=300
            )
        )
        return response.text
    
    def execute(self, process_log: ProcessLog, data_directory: str) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "resource_pooling", "Starting resource analysis", AgentStatus.RUNNING)
        
        results = {
            "pdf_overviews": {},
            "csv_overviews": {},
            "extracted_csvs": {},
            "cached_files": list(self.cached_files.keys())
        }
        
        data_path = Path(data_directory)
        
        for file_path in data_path.glob("**/*"):
            if file_path.is_file():
                if file_path.suffix.lower() == '.pdf':
                    overview = self._analyze_pdf_overview(str(file_path))
                    results["pdf_overviews"][str(file_path)] = overview
                
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    csv_files = self._extract_sheets_to_csv(str(file_path))
                    results["extracted_csvs"][str(file_path)] = csv_files
                    
                    for sheet_name, csv_path in csv_files.items():
                        overview = self._analyze_csv_overview(csv_path)
                        results["csv_overviews"][csv_path] = overview
        
        process_log.log(self.__class__.__name__, "resource_pooling", results, AgentStatus.COMPLETED)
        return results

class ResourcePoolerCheckerAgent(BaseAgent):
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "resource_verification", "Verifying resource processing", AgentStatus.RUNNING)
        
        resource_data = process_log.get_stage_data("resource_pooling")
        
        if not resource_data:
            process_log.log(self.__class__.__name__, "resource_verification", "No resource data found", AgentStatus.FAILED)
            return {"verified": False, "issues": ["No resource pooling data found"]}
        
        issues = []
        
        if not resource_data.get("pdf_overviews") and not resource_data.get("csv_overviews"):
            issues.append("No documents processed")
        
        if resource_data.get("extracted_csvs"):
            for excel_file, csv_files in resource_data["extracted_csvs"].items():
                if not csv_files:
                    issues.append(f"No sheets extracted from {excel_file}")
        
        verification_result = {
            "verified": len(issues) == 0,
            "issues": issues,
            "total_pdfs": len(resource_data.get("pdf_overviews", {})),
            "total_csvs": len(resource_data.get("csv_overviews", {})),
            "total_excel_files": len(resource_data.get("extracted_csvs", {}))
        }
        
        status = AgentStatus.VERIFIED if verification_result["verified"] else AgentStatus.FAILED
        process_log.log(self.__class__.__name__, "resource_verification", verification_result, status)
        
        return verification_result

class AnalystAgent(BaseAgent):
    def _generate_analysis_questions(self, document_overviews: Dict[str, str]) -> List[str]:
        overview_text = "\n".join([f"{file}: {overview}" for file, overview in document_overviews.items()])
        
        prompt = f"""
        Based on these document overviews for investment analysis, generate 8-10 specific questions that would help assess:
        1. Financial performance and stability
        2. Business model and revenue streams  
        3. Market position and competitive advantages
        4. Risk factors and challenges
        5. Growth prospects and scalability
        
        Document Overviews:
        {overview_text}
        
        Generate questions that can be answered from the documents. Format as a numbered list.
        """
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=800
            )
        )
        
        questions = []
        for line in response.text.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                questions.append(line.strip())
        
        return questions
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "analysis", "Starting document analysis", AgentStatus.RUNNING)
        
        resource_data = process_log.get_stage_data("resource_pooling")
        verification_data = process_log.get_stage_data("resource_verification")
        
        if not verification_data or not verification_data.get("verified"):
            process_log.log(self.__class__.__name__, "analysis", "Resource verification failed", AgentStatus.FAILED)
            return {"error": "Cannot proceed without verified resources"}
        
        all_overviews = {}
        all_overviews.update(resource_data.get("pdf_overviews", {}))
        all_overviews.update(resource_data.get("csv_overviews", {}))
        
        questions = self._generate_analysis_questions(all_overviews)
        
        analysis_results = {
            "generated_questions": questions,
            "document_insights": {},
            "key_findings": []
        }
        
        process_log.log(self.__class__.__name__, "analysis", analysis_results, AgentStatus.COMPLETED)
        return analysis_results

class AnalystCheckerAgent(BaseAgent):
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "analyst_verification", "Verifying analyst outputs", AgentStatus.RUNNING)
        
        analysis_data = process_log.get_stage_data("analysis")
        
        if not analysis_data:
            process_log.log(self.__class__.__name__, "analyst_verification", "No analysis data found", AgentStatus.FAILED)
            return {"verified": False, "issues": ["No analysis data found"]}
        
        prompt = f"""
        Evaluate the quality and relevance of this investment analysis output:
        
        Generated Questions: {len(analysis_data.get('generated_questions', []))}
        Sample Questions: {analysis_data.get('generated_questions', [])[:3]}
        
        Assessment Criteria:
        1. Are questions relevant for investment decision-making?
        2. Do questions cover key financial assessment areas?
        3. Are questions specific enough to extract actionable insights?
        4. Quality score (1-10) and brief explanation
        
        Respond in JSON format with: {{"quality_score": X, "relevant": true/false, "issues": ["list"], "recommendation": "proceed/revise"}}
        """
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=400
            )
        )
        
        try:
            verification_result = json.loads(response.text.strip())
            verification_result["verified"] = verification_result.get("quality_score", 0) >= 7
        except:
            verification_result = {"verified": False, "issues": ["Failed to parse verification response"]}
        
        status = AgentStatus.VERIFIED if verification_result["verified"] else AgentStatus.FAILED
        process_log.log(self.__class__.__name__, "analyst_verification", verification_result, status)
        
        return verification_result

class AssociateAgent(BaseAgent):
    def _calculate_financial_ratios(self, csv_data: Dict[str, str]) -> Dict[str, Any]:
        ratio_analyses = {}
        
        for csv_path, overview in csv_data.items():
            try:
                df = pd.read_csv(csv_path)
                
                prompt = f"""
                Analyze this financial data and calculate relevant investment ratios:
                
                Dataset Overview: {overview}
                Columns: {df.columns.tolist()}
                Sample Data: {df.head(2).to_string()}
                
                Identify and calculate (where possible):
                1. Liquidity ratios (Current Ratio, Quick Ratio)
                2. Profitability ratios (ROE, ROA, Profit Margins)
                3. Leverage ratios (Debt-to-Equity, Interest Coverage)
                4. Efficiency ratios (Asset Turnover, Inventory Turnover)
                5. Any suspicious or noteworthy figures
                
                Provide specific calculations and flag any concerning trends.
                Format as structured analysis with clear findings.
                """
                
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=600
                    )
                )
                
                ratio_analyses[csv_path] = {
                    "analysis": response.text,
                    "data_shape": df.shape,
                    "columns": df.columns.tolist()
                }
                
            except Exception as e:
                ratio_analyses[csv_path] = {"error": str(e)}
        
        return ratio_analyses
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "financial_analysis", "Starting financial ratio analysis", AgentStatus.RUNNING)
        
        resource_data = process_log.get_stage_data("resource_pooling")
        analyst_verification = process_log.get_stage_data("analyst_verification")
        
        if not analyst_verification or not analyst_verification.get("verified"):
            process_log.log(self.__class__.__name__, "financial_analysis", "Analyst verification failed", AgentStatus.FAILED)
            return {"error": "Cannot proceed without verified analyst output"}
        
        csv_overviews = resource_data.get("csv_overviews", {})
        ratio_analyses = self._calculate_financial_ratios(csv_overviews)
        
        associate_results = {
            "ratio_analyses": ratio_analyses,
            "key_financial_insights": [],
            "red_flags": [],
            "investment_metrics": {}
        }
        
        process_log.log(self.__class__.__name__, "financial_analysis", associate_results, AgentStatus.COMPLETED)
        return associate_results

class SectorSpecialistAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.search_tool = Tool(google_search=GoogleSearch())
    
    def _research_sector_comparisons(self, company_context: str) -> Dict[str, Any]:
        search_queries = [
            f"NBA sector India financial performance 2024 {company_context}",
            f"performing credit companies India ratios benchmarks",
            f"alternative investment fund category 2 India portfolio companies performance",
            f"NBA sector growth prospects India market analysis"
        ]
        
        research_results = {}
        
        for query in search_queries:
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=f"Research and analyze: {query}. Focus on financial metrics, performance indicators, and market position of companies in this sector.",
                    config=GenerateContentConfig(
                        tools=[self.search_tool],
                        temperature=0.4,
                        max_output_tokens=500
                    )
                )
                
                research_results[query] = {
                    "findings": response.text,
                    "timestamp": datetime.now().isoformat()
                }
                
                time.sleep(1)
                
            except Exception as e:
                research_results[query] = {"error": str(e)}
        
        return research_results
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "sector_research", "Starting sector analysis", AgentStatus.RUNNING)
        
        associate_data = process_log.get_stage_data("financial_analysis")
        resource_data = process_log.get_stage_data("resource_pooling")
        
        if not associate_data:
            process_log.log(self.__class__.__name__, "sector_research", "No associate data found", AgentStatus.FAILED)
            return {"error": "Cannot proceed without associate analysis"}
        
        company_context = "NBA sector financial services"
        research_results = self._research_sector_comparisons(company_context)
        
        sector_analysis = {
            "market_research": research_results,
            "competitive_analysis": {},
            "sector_benchmarks": {},
            "market_trends": []
        }
        
        process_log.log(self.__class__.__name__, "sector_research", sector_analysis, AgentStatus.COMPLETED)
        return sector_analysis

class SeniorAgent(BaseAgent):
    def _generate_investment_recommendation(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        resource_summary = all_data.get("resource_pooling", {})
        financial_analysis = all_data.get("financial_analysis", {})
        sector_research = all_data.get("sector_research", {})
        
        prompt = f"""
        As a senior investment analyst for an AIF Category 2 fund, provide a comprehensive investment recommendation based on this analysis:
        
        RESOURCE ANALYSIS:
        - PDFs Processed: {len(resource_summary.get('pdf_overviews', {}))}
        - CSVs Analyzed: {len(resource_summary.get('csv_overviews', {}))}
        
        FINANCIAL ANALYSIS:
        - Ratio Analyses Completed: {len(financial_analysis.get('ratio_analyses', {}))}
        
        SECTOR RESEARCH:
        - Market Research Queries: {len(sector_research.get('market_research', {}))}
        
        Provide a structured recommendation including:
        1. EXECUTIVE SUMMARY (2-3 sentences)
        2. KEY STRENGTHS (bullet points)
        3. MAJOR CONCERNS/RISKS (bullet points)  
        4. FINANCIAL HEALTH ASSESSMENT (1-10 score)
        5. SECTOR POSITION ANALYSIS
        6. INVESTMENT RECOMMENDATION (INVEST/HOLD/PASS with rationale)
        7. SUGGESTED DUE DILIGENCE ACTIONS
        8. HUMAN REVIEW PRIORITIES
        
        Be specific about what human review should focus on.
        """
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000
            )
        )
        
        return {
            "investment_recommendation": response.text,
            "requires_human_review": True,
            "review_priority": "HIGH",
            "generated_at": datetime.now().isoformat()
        }
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "final_assessment", "Generating final investment recommendation", AgentStatus.RUNNING)
        
        all_stage_data = {}
        for entry in process_log.entries:
            if entry["status"] == "completed":
                all_stage_data[entry["stage"]] = entry["data"]
        
        final_recommendation = self._generate_investment_recommendation(all_stage_data)
        
        senior_results = {
            "final_recommendation": final_recommendation,
            "process_summary": {
                "total_stages": len(all_stage_data),
                "total_duration": (datetime.now() - process_log.start_time).total_seconds(),
                "status": "READY_FOR_HUMAN_REVIEW"
            }
        }
        
        process_log.log(self.__class__.__name__, "final_assessment", senior_results, AgentStatus.COMPLETED)
        return senior_results

class MetaAgent:
    def __init__(self):
        self.process_log = ProcessLog()
        self.agents = {
            "resource_pooler": ResourcePoolerAgent(),
            "resource_pooler_checker": ResourcePoolerCheckerAgent(),
            "analyst": AnalystAgent(),
            "analyst_checker": AnalystCheckerAgent(),
            "associate": AssociateAgent(),
            "sector_specialist": SectorSpecialistAgent(),
            "senior": SeniorAgent()
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agentic/process.log'),
                logging.StreamHandler()
            ]
        )
    
    def execute_pipeline(self, data_directory: str) -> Dict[str, Any]:
        try:
            # Stage 1: Resource Pooling
            resource_result = self.agents["resource_pooler"].execute(self.process_log, data_directory=data_directory)
            
            # Stage 2: Resource Verification
            verification_result = self.agents["resource_pooler_checker"].execute(self.process_log)
            if not verification_result.get("verified"):
                return {
                    "status": "failed",
                    "stage": "resource_verification", 
                    "issues": verification_result.get("issues", [])
                }
            
            # Stage 3: Analyst Analysis
            analysis_result = self.agents["analyst"].execute(self.process_log)
            
            # Stage 4: Analyst Verification
            analyst_verification = self.agents["analyst_checker"].execute(self.process_log)
            if not analyst_verification.get("verified"):
                return {
                    "status": "failed",
                    "stage": "analyst_verification",
                    "issues": analyst_verification.get("issues", [])
                }
            
            # Stage 5: Associate Financial Analysis
            associate_result = self.agents["associate"].execute(self.process_log)
            
            # Stage 6: Sector Research
            sector_result = self.agents["sector_specialist"].execute(self.process_log)
            
            # Stage 7: Senior Assessment
            senior_result = self.agents["senior"].execute(self.process_log)
            
            return {
                "status": "completed",
                "process_log": self.process_log.entries,
                "final_results": {
                    "resource_pooling": resource_result,
                    "verification": verification_result,
                    "analysis": analysis_result,
                    "analyst_verification": analyst_verification,
                    "financial_analysis": associate_result,
                    "sector_research": sector_result,
                    "final_assessment": senior_result
                },
                "human_review_required": True,
                "next_actions": senior_result.get("process_summary", {})
            }
            
        except Exception as e:
            self.process_log.log("MetaAgent", "pipeline_execution", str(e), AgentStatus.FAILED)
            return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    load_dotenv()
    
    meta_agent = MetaAgent()
    result = meta_agent.execute_pipeline("data/")
    
    print(json.dumps(result, indent=2, default=str))