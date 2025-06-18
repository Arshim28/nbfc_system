import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from agentic.base.base_agent import ProcessLog, AgentStatus
from maker_agents.resource_pooler import ResourcePoolerAgent
from checker_agents.resource_pooler_checker import ResourcePoolerCheckerAgent
from maker_agents.analyst import AnalystAgent
from checker_agents.analyst_checker import AnalystCheckerAgent
from maker_agents.associate import AssociateAgent
from maker_agents.sector_specialist import SectorSpecialistAgent
from maker_agents.senior import SeniorAgent

load_dotenv()

class MetaAgent:
    def __init__(self, config_path: str = "agentic/config.yaml"):
        self.process_log = ProcessLog()
        self.config_path = config_path
        self.setup_logging()
        
        # Initialize all agents
        self.agents = {
            "resource_pooler": ResourcePoolerAgent(),
            "resource_pooler_checker": ResourcePoolerCheckerAgent(),
            "analyst": AnalystAgent(),
            "analyst_checker": AnalystCheckerAgent(),
            "associate": AssociateAgent(),
            "sector_specialist": SectorSpecialistAgent(),
            "senior": SeniorAgent()
        }
        
        # Pipeline configuration
        self.pipeline_stages = [
            {
                "name": "document_harvest",
                "agent": "resource_pooler",
                "description": "Document harvest & structuring",
                "required_params": ["data_directory"],
                "timeout_minutes": 15,
                "retry_count": 1
            },
            {
                "name": "ingestion_qa", 
                "agent": "resource_pooler_checker",
                "description": "Resource verification & QA",
                "dependencies": ["document_harvest"],
                "timeout_minutes": 5,
                "retry_count": 2
            },
            {
                "name": "qualitative_quantitative_inquiry",
                "agent": "analyst",
                "description": "Analyst investigation & document analysis", 
                "dependencies": ["ingestion_qa"],
                "timeout_minutes": 20,
                "retry_count": 1
            },
            {
                "name": "analyst_verification",
                "agent": "analyst_checker", 
                "description": "Analyst output verification",
                "dependencies": ["qualitative_quantitative_inquiry"],
                "timeout_minutes": 5,
                "retry_count": 2
            },
            {
                "name": "financial_ratio_analysis",
                "agent": "associate",
                "description": "Financial ratio deep-dive analysis",
                "dependencies": ["analyst_verification"],
                "timeout_minutes": 15,
                "retry_count": 1
            },
            {
                "name": "sector_research",
                "agent": "sector_specialist",
                "description": "External benchmark & macro analysis",
                "dependencies": ["financial_ratio_analysis"],
                "timeout_minutes": 25,
                "retry_count": 1
            },
            {
                "name": "ic_synthesis",
                "agent": "senior",
                "description": "IC-level synthesis & risk-return",
                "dependencies": ["sector_research"],
                "timeout_minutes": 10,
                "retry_count": 1
            }
        ]
        
        self.logger = logging.getLogger("MetaAgent")
    
    def setup_logging(self):
        log_dir = Path("agentic/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_filename = f"gold_loan_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    
    def _validate_dependencies(self, stage: Dict[str, Any]) -> bool:
        if "dependencies" not in stage:
            return True
        
        for dependency in stage["dependencies"]:
            dependency_data = self.process_log.get_stage_data(dependency)
            if not dependency_data:
                self.logger.error(f"Dependency {dependency} not satisfied for stage {stage['name']}")
                return False
            
            # Check if dependency was verified (for checker agents)
            if dependency.endswith("_verification") or dependency.endswith("_qa"):
                if not dependency_data.get("verified", False):
                    self.logger.error(f"Dependency {dependency} failed verification")
                    return False
        
        return True
    
    def _execute_stage_with_retry(self, stage: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        stage_name = stage["name"]
        agent_name = stage["agent"]
        max_retries = stage.get("retry_count", 1)
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Executing stage {stage_name} (attempt {attempt + 1}/{max_retries + 1})")
                
                agent = self.agents[agent_name]
                
                # Execute the agent
                if stage_name == "document_harvest":
                    result = agent.execute(self.process_log, **kwargs)
                else:
                    result = agent.execute(self.process_log)
                
                # Check if execution was successful
                if "error" in result:
                    raise Exception(f"Agent returned error: {result['error']}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Stage {stage_name} attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries:
                    self.process_log.log(
                        "MetaAgent", 
                        stage_name, 
                        {"error": str(e), "attempts": attempt + 1}, 
                        AgentStatus.FAILED
                    )
                    raise Exception(f"Stage {stage_name} failed after {attempt + 1} attempts: {str(e)}")
                
                # Wait before retry
                import time
                time.sleep(5 * (attempt + 1))
    
    def _validate_stage_output(self, stage: Dict[str, Any], result: Dict[str, Any]) -> bool:
        stage_name = stage["name"]
        
        # Basic validation - check for required fields
        if "error" in result:
            self.logger.error(f"Stage {stage_name} returned error: {result['error']}")
            return False
        
        # Stage-specific validation
        if stage_name == "document_harvest":
            required_fields = ["pdf_analyses", "csv_analyses", "processing_summary"]
            if not all(field in result for field in required_fields):
                self.logger.error(f"Stage {stage_name} missing required fields")
                return False
            
            if result["processing_summary"]["total_files_processed"] == 0:
                self.logger.error("No files were processed in document harvest")
                return False
        
        elif stage_name in ["ingestion_qa", "analyst_verification"]:
            if not result.get("verified", False):
                self.logger.warning(f"Stage {stage_name} verification failed")
                return False
        
        elif stage_name == "qualitative_quantitative_inquiry":
            if len(result.get("key_findings", [])) < 3:
                self.logger.warning("Insufficient key findings from analyst")
                # Don't fail, but log warning
        
        elif stage_name == "financial_ratio_analysis":
            if not result.get("ratio_analyses"):
                self.logger.error("No ratio analyses completed")
                return False
        
        elif stage_name == "sector_research":
            research_quality = result.get("research_quality", {})
            if research_quality.get("queries_completed", 0) < 5:
                self.logger.warning("Limited sector research completed")
                # Don't fail, but log warning
        
        elif stage_name == "ic_synthesis":
            if not result.get("ic_memorandum"):
                self.logger.error("IC memorandum not generated")
                return False
        
        return True
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        completed_stages = []
        failed_stages = []
        
        for entry in self.process_log.entries:
            if entry["status"] == "completed":
                completed_stages.append(entry["stage"])
            elif entry["status"] == "failed":
                failed_stages.append(entry["stage"])
        
        total_duration = (datetime.now() - self.process_log.start_time).total_seconds()
        
        return {
            "pipeline_status": "COMPLETED" if len(failed_stages) == 0 else "FAILED",
            "completed_stages": completed_stages,
            "failed_stages": failed_stages,
            "total_stages": len(self.pipeline_stages),
            "completion_rate": len(completed_stages) / len(self.pipeline_stages),
            "total_duration_minutes": round(total_duration / 60, 2),
            "started_at": self.process_log.start_time.isoformat(),
            "completed_at": datetime.now().isoformat()
        }
    
    def _save_results(self, results: Dict[str, Any], data_directory: str):
        output_dir = Path(data_directory) / "analysis_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        results_file = output_dir / f"gold_loan_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save process log separately
        log_file = output_dir / f"process_log_{timestamp}.json"
        self.process_log.save_to_file(str(log_file))
        
        # Save IC memorandum separately for easy access
        if "final_results" in results and "ic_synthesis" in results["final_results"]:
            ic_memo = results["final_results"]["ic_synthesis"].get("ic_memorandum", {})
            if ic_memo:
                memo_file = output_dir / f"ic_memorandum_{timestamp}.txt"
                with open(memo_file, 'w') as f:
                    f.write(ic_memo.get("ic_memorandum", ""))
        
        self.logger.info(f"Results saved to {output_dir}")
        return str(results_file)
    
    def execute_pipeline(self, data_directory: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Execute the complete gold loan NBFC investment analysis pipeline.
        
        Args:
            data_directory: Path to directory containing PDF and Excel files
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing pipeline results and summary
        """
        self.logger.info(f"Starting gold loan NBFC analysis pipeline for: {data_directory}")
        self.process_log.log("MetaAgent", "pipeline_start", f"Data directory: {data_directory}", AgentStatus.RUNNING)
        
        # Validate data directory
        if not Path(data_directory).exists():
            error_msg = f"Data directory does not exist: {data_directory}"
            self.logger.error(error_msg)
            return {"status": "failed", "error": error_msg}
        
        pipeline_results = {"final_results": {}}
        
        try:
            # Execute each stage in sequence
            for stage in self.pipeline_stages:
                stage_name = stage["name"]
                
                self.logger.info(f"Starting stage: {stage_name} - {stage['description']}")
                
                # Validate dependencies
                if not self._validate_dependencies(stage):
                    error_msg = f"Dependencies not satisfied for stage {stage_name}"
                    self.process_log.log("MetaAgent", stage_name, error_msg, AgentStatus.FAILED)
                    return {
                        "status": "failed",
                        "stage": stage_name,
                        "error": error_msg,
                        "pipeline_summary": self._generate_pipeline_summary()
                    }
                
                # Execute stage with retry logic
                try:
                    if stage_name == "document_harvest":
                        stage_result = self._execute_stage_with_retry(stage, data_directory=data_directory)
                    else:
                        stage_result = self._execute_stage_with_retry(stage)
                    
                    # Validate output
                    if not self._validate_stage_output(stage, stage_result):
                        error_msg = f"Stage {stage_name} output validation failed"
                        return {
                            "status": "failed",
                            "stage": stage_name,
                            "error": error_msg,
                            "pipeline_summary": self._generate_pipeline_summary()
                        }
                    
                    pipeline_results["final_results"][stage_name] = stage_result
                    self.logger.info(f"Stage {stage_name} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Stage {stage_name} execution failed: {str(e)}"
                    self.logger.error(error_msg)
                    return {
                        "status": "failed",
                        "stage": stage_name,
                        "error": error_msg,
                        "pipeline_summary": self._generate_pipeline_summary()
                    }
            
            # Pipeline completed successfully
            pipeline_summary = self._generate_pipeline_summary()
            
            final_results = {
                "status": "completed",
                "pipeline_summary": pipeline_summary,
                "final_results": pipeline_results["final_results"],
                "process_log_entries": self.process_log.entries,
                "human_review_required": True,
                "ic_ready": True,
                "next_actions": {
                    "immediate": "Review IC memorandum and schedule Investment Committee presentation",
                    "follow_up": "Execute human due diligence as prioritized by Senior Agent",
                    "timeline": "IC decision target: 6-8 weeks"
                }
            }
            
            # Save results if requested
            if save_results:
                results_file = self._save_results(final_results, data_directory)
                final_results["results_saved_to"] = results_file
            
            self.process_log.log(
                "MetaAgent", 
                "pipeline_completion", 
                pipeline_summary, 
                AgentStatus.COMPLETED,
                f"All {len(self.pipeline_stages)} stages completed successfully"
            )
            
            return final_results
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "status": "failed",
                "error": error_msg,
                "pipeline_summary": self._generate_pipeline_summary(),
                "partial_results": pipeline_results.get("final_results", {})
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status."""
        return {
            "current_stage": self.process_log.current_stage,
            "elapsed_time_minutes": (datetime.now() - self.process_log.start_time).total_seconds() / 60,
            "completed_stages": [entry["stage"] for entry in self.process_log.entries if entry["status"] == "completed"],
            "failed_stages": [entry["stage"] for entry in self.process_log.entries if entry["status"] == "failed"],
            "total_entries": len(self.process_log.entries)
        }

# Convenience function for running the pipeline
def run_gold_loan_analysis(data_directory: str, config_path: str = "agentic/config.yaml") -> Dict[str, Any]:
    """
    Convenience function to run the complete gold loan NBFC analysis pipeline.
    
    Args:
        data_directory: Path to directory containing analysis documents
        config_path: Path to configuration file
        
    Returns:
        Pipeline execution results
    """
    meta_agent = MetaAgent(config_path)
    return meta_agent.execute_pipeline(data_directory)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/meta_agent.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    result = run_gold_loan_analysis(data_dir)
    print("\n" + "="*50)
    print("GOLD LOAN NBFC ANALYSIS PIPELINE RESULTS")
    print("="*50)
    print(f"Status: {result['status'].upper()}")
    if result['status'] == 'completed':
        summary = result['pipeline_summary']
        print(f"Stages Completed: {summary['completed_stages']}")
        print(f"Total Duration: {summary['total_duration_minutes']} minutes")
        print(f"IC Ready: {result.get('ic_ready', False)}")
        print(f"Results Saved: {result.get('results_saved_to', 'Not saved')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Failed at stage: {result.get('stage', 'Unknown')}")
    print("="*50)