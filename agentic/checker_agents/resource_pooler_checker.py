import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel, Field
from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

class DocumentMetadata(BaseModel):
    """Pydantic model for validating document metadata structure"""
    model_config = {"extra": "allow"}  # Allow additional fields
    
    name: str
    summary: str
    analyst_info: str
    file_path: str
    cache_name: str
    file_id: str | None = None  # Allow None values
    token_usage: Dict[str, int] = Field(default_factory=dict)
    reused_cache: bool = False

class ResourcePoolerCheckerAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.logger = logging.getLogger("ResourcePoolerCheckerAgent")
        self.log_file = Path("agentic/logs/resource_pooler.log")
        self.required_document_types = [
            "annual_report", "financial_statements", "debenture_trust_deed", 
            "portfolio_data", "operations_data", "alm_data", "regulatory_circular"
        ]
    
    def _validate_metadata_structure(self, metadata_entries: List[Dict]) -> Dict[str, Any]:
        """Validate that all metadata entries have the required structure"""
        issues = []
        valid_entries = 0
        pydantic_validation_errors = []
        
        for i, entry in enumerate(metadata_entries):
            try:
                # Validate with Pydantic model
                DocumentMetadata(**entry)
                valid_entries += 1
            except Exception as e:
                pydantic_validation_errors.append(f"Entry {i+1} ({entry.get('name', 'unknown')}): {str(e)}")
                issues.append(f"Invalid metadata structure in entry {i+1}: {entry.get('name', 'unknown')}")
        
        return {
            "valid_entries": valid_entries,
            "total_entries": len(metadata_entries),
            "validation_success_rate": (valid_entries / len(metadata_entries)) * 100 if metadata_entries else 0,
            "pydantic_errors": pydantic_validation_errors,
            "issues": issues
        }
    
    def _analyze_document_types(self, metadata_entries: List[Dict]) -> Dict[str, Any]:
        """Analyze the types of documents processed and their coverage"""
        document_types = defaultdict(list)
        file_extensions = defaultdict(int)
        
        for entry in metadata_entries:
            file_path = entry.get("file_path", "")
            file_name = entry.get("name", "")
            
            # Categorize by file extension
            if file_path:
                ext = Path(file_path).suffix.lower()
                file_extensions[ext] += 1
            
            # Categorize by document type based on content
            summary = entry.get("summary", "").lower()
            analyst_info = entry.get("analyst_info", "").lower()
            content = f"{summary} {analyst_info} {file_name.lower()}"
            
            if any(term in content for term in ["annual report", "annual"]):
                document_types["annual_reports"].append(file_name)
            elif any(term in content for term in ["debenture", "trust deed", "dtd"]):
                document_types["debenture_documents"].append(file_name)
            elif any(term in content for term in ["portfolio", "loan", "par", "dpd"]):
                document_types["portfolio_data"].append(file_name)
            elif any(term in content for term in ["financial", "balance sheet", "profit", "cash flow"]):
                document_types["financial_statements"].append(file_name)
            elif any(term in content for term in ["alm", "asset liability", "interest rate"]):
                document_types["alm_data"].append(file_name)
            elif any(term in content for term in ["operations", "branch", "borrower"]):
                document_types["operations_data"].append(file_name)
            elif any(term in content for term in ["circular", "rbi", "regulatory"]):
                document_types["regulatory_documents"].append(file_name)
            else:
                document_types["other"].append(file_name)
        
        return {
            "document_types": dict(document_types),
            "file_extensions": dict(file_extensions),
            "type_coverage": len([dt for dt in document_types.keys() if dt != "other"]),
            "total_categories": len(self.required_document_types)
        }
    
    def _analyze_cache_performance(self, metadata_entries: List[Dict]) -> Dict[str, Any]:
        """Analyze cache reuse performance and token usage"""
        total_entries = len(metadata_entries)
        reused_caches = sum(1 for entry in metadata_entries if entry.get("reused_cache", False))
        new_caches = total_entries - reused_caches
        
        total_tokens = 0
        prompt_tokens = 0
        candidate_tokens = 0
        
        for entry in metadata_entries:
            token_usage = entry.get("token_usage", {})
            total_tokens += token_usage.get("total", 0)
            prompt_tokens += token_usage.get("prompt", 0)
            candidate_tokens += token_usage.get("candidates", 0)
        
        cache_reuse_rate = (reused_caches / total_entries) * 100 if total_entries > 0 else 0
        
        return {
            "total_documents": total_entries,
            "reused_caches": reused_caches,
            "new_caches": new_caches,
            "cache_reuse_rate": cache_reuse_rate,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "avg_tokens_per_document": total_tokens / total_entries if total_entries > 0 else 0
        }
    
    def _check_content_quality(self, metadata_entries: List[Dict]) -> Dict[str, Any]:
        """Check the quality and completeness of generated content"""
        quality_issues = []
        high_quality_entries = 0
        
        for entry in metadata_entries:
            name = entry.get("name", "unknown")
            summary = entry.get("summary", "")
            analyst_info = entry.get("analyst_info", "")
            
            # Check for minimum content length
            if len(summary) < 50:
                quality_issues.append(f"{name}: Summary too short ({len(summary)} chars)")
            elif len(summary) > 1000:
                quality_issues.append(f"{name}: Summary too long ({len(summary)} chars)")
            
            if len(analyst_info) < 50:
                quality_issues.append(f"{name}: Analyst info too short ({len(analyst_info)} chars)")
            elif len(analyst_info) > 1000:
                quality_issues.append(f"{name}: Analyst info too long ({len(analyst_info)} chars)")
            
            # Check for generic/fallback content
            if "processing error" in summary.lower() or "formatting failed" in summary.lower():
                quality_issues.append(f"{name}: Contains error/fallback content")
            elif "fallback_used" in entry:
                quality_issues.append(f"{name}: Used fallback generation")
            else:
                high_quality_entries += 1
        
        quality_score = (high_quality_entries / len(metadata_entries)) * 100 if metadata_entries else 0
        
        return {
            "high_quality_entries": high_quality_entries,
            "total_entries": len(metadata_entries),
            "quality_score": quality_score,
            "quality_issues": quality_issues,
            "meets_quality_threshold": quality_score >= 80
        }
    
    def _check_file_coverage(self, data_directory: str, metadata_entries: List[Dict]) -> Dict[str, Any]:
        """Check if all files in the data directory are covered in the metadata"""
        data_path = Path(data_directory)
        
        # Get all files in data directory (excluding extracted_sheets directory)
        all_files = set()
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and "extracted_sheets" not in str(file_path):
                all_files.add(str(file_path.resolve()))
        
        # Get files from metadata
        logged_files = set()
        extracted_sheet_files = set()
        
        for entry in metadata_entries:
            file_path = entry.get("file_path", "")
            if file_path:
                abs_path = str(Path(file_path).resolve())
                if "extracted_sheets" in file_path:
                    extracted_sheet_files.add(abs_path)
                else:
                    logged_files.add(abs_path)
        
        # Find missed files
        missed_files = sorted(list(all_files - logged_files))
        coverage_rate = ((len(all_files) - len(missed_files)) / len(all_files)) * 100 if all_files else 100
        
        return {
            "total_data_files": len(all_files),
            "covered_files": len(logged_files),
            "extracted_sheet_files": len(extracted_sheet_files),
            "missed_files": missed_files,
            "coverage_rate": coverage_rate,
            "all_files_covered": len(missed_files) == 0
        }

    def execute(self, process_log: ProcessLog, data_directory: str) -> None:
        self.logger.info(f"Starting ResourcePoolerChecker for data_directory: {data_directory}")
        process_log.log(self.__class__.__name__, "resource_pooler_check", "Starting resource pooler validation", AgentStatus.RUNNING)
        
        # Read metadata entries from log file
        metadata_entries = []
        if not self.log_file.exists():
            self.logger.error(f"Resource pooler log file not found: {self.log_file}")
            process_log.log(self.__class__.__name__, "resource_pooler_check", 
                          {"error": "Log file not found"}, AgentStatus.FAILED)
            return
        
        try:
            with open(self.log_file, "r") as log_f:
                for line_num, line in enumerate(log_f, 1):
                    try:
                        entry = json.loads(line.strip())
                        metadata_entries.append(entry)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing line {line_num}: {e}")
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
            process_log.log(self.__class__.__name__, "resource_pooler_check", 
                          {"error": f"Failed to read log file: {e}"}, AgentStatus.FAILED)
            return
        
        if not metadata_entries:
            self.logger.warning("No metadata entries found in log file")
            process_log.log(self.__class__.__name__, "resource_pooler_check", 
                          {"warning": "No metadata entries found"}, AgentStatus.COMPLETED)
            return
        
        # Perform various validation checks
        self.logger.info(f"Validating {len(metadata_entries)} metadata entries")
        
        structure_validation = self._validate_metadata_structure(metadata_entries)
        document_analysis = self._analyze_document_types(metadata_entries)
        cache_performance = self._analyze_cache_performance(metadata_entries)
        content_quality = self._check_content_quality(metadata_entries)
        file_coverage = self._check_file_coverage(data_directory, metadata_entries)
        
        # Compile overall results
        results = {
            "total_entries": len(metadata_entries),
            "structure_validation": structure_validation,
            "document_analysis": document_analysis,
            "cache_performance": cache_performance,
            "content_quality": content_quality,
            "file_coverage": file_coverage,
            "overall_health": {
                "structure_valid": structure_validation["validation_success_rate"] >= 95,
                "content_quality_good": content_quality["meets_quality_threshold"],
                "full_file_coverage": file_coverage["all_files_covered"],
                "cache_performance_good": cache_performance["cache_reuse_rate"] >= 50
            }
        }
        
        # Log summary
        self.logger.info("=== RESOURCE POOLER VALIDATION SUMMARY ===")
        self.logger.info(f"Total entries processed: {len(metadata_entries)}")
        self.logger.info(f"Structure validation: {structure_validation['validation_success_rate']:.1f}%")
        self.logger.info(f"Content quality score: {content_quality['quality_score']:.1f}%")
        self.logger.info(f"File coverage: {file_coverage['coverage_rate']:.1f}%")
        self.logger.info(f"Cache reuse rate: {cache_performance['cache_reuse_rate']:.1f}%")
        self.logger.info(f"Total token usage: {cache_performance['total_tokens']:,}")
        
        # Print detailed results
        print("\n" + "="*60)
        print("RESOURCE POOLER VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  • Total entries: {len(metadata_entries)}")
        print(f"  • Structure validation: {structure_validation['validation_success_rate']:.1f}%")
        print(f"  • Content quality: {content_quality['quality_score']:.1f}%")
        print(f"  • File coverage: {file_coverage['coverage_rate']:.1f}%")
        
        print(f"\n🔄 CACHE PERFORMANCE:")
        print(f"  • Cache reuse rate: {cache_performance['cache_reuse_rate']:.1f}%")
        print(f"  • Reused caches: {cache_performance['reused_caches']}")
        print(f"  • New caches: {cache_performance['new_caches']}")
        print(f"  • Total tokens: {cache_performance['total_tokens']:,}")
        
        print(f"\n📁 DOCUMENT TYPES:")
        for doc_type, files in document_analysis['document_types'].items():
            print(f"  • {doc_type}: {len(files)} files")
        
        if content_quality['quality_issues']:
            print(f"\n⚠️  QUALITY ISSUES ({len(content_quality['quality_issues'])}):")
            for issue in content_quality['quality_issues'][:5]:  # Show first 5
                print(f"  • {issue}")
            if len(content_quality['quality_issues']) > 5:
                print(f"  • ... and {len(content_quality['quality_issues']) - 5} more")
        
        if file_coverage['missed_files']:
            print(f"\n❌ MISSED FILES ({len(file_coverage['missed_files'])}):")
            for missed_file in file_coverage['missed_files']:
                print(f"  • {missed_file}")
        else:
            print(f"\n✅ ALL FILES COVERED")
        
        overall_health = results['overall_health']
        all_good = all(overall_health.values())
        
        print(f"\n{'✅' if all_good else '⚠️ '} OVERALL HEALTH: {'EXCELLENT' if all_good else 'NEEDS ATTENTION'}")
        for check, status in overall_health.items():
            print(f"  • {check.replace('_', ' ').title()}: {'✅' if status else '❌'}")
        
        print("="*60)
        
        process_log.log(self.__class__.__name__, "resource_pooler_check", results, AgentStatus.COMPLETED)

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/checker_agents/resource_pooler_checker.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    from agentic.base.base_agent import ProcessLog
    process_log = ProcessLog()
    agent = ResourcePoolerCheckerAgent()
    agent.execute(process_log, data_dir)