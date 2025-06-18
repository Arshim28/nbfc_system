from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

class AnalystAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.analysis_questions = {
            "business_strategy": [
                "How has the share of non-gold collateral products evolved (% of AUM) FY21-FY25?",
                "What CAGR of branch additions did management target vs. actual?",
                "Does the strategic roadmap mention digital gold loans / door-step service adoption metrics?"
            ],
            "asset_quality": [
                "Static-pool Stage-3 for FY22 originations at 12M, 24M, 36M vintages?",
                "What % of restructured gold loans (under ECLGS) slipped into Stage-3 within 12M?", 
                "Has auditor highlighted any ECL model override ≥ ₹25 cr?"
            ],
            "underwriting_risk": [
                "Current weighted-average LTV vs. RBI cap (75%); distribution by ticket-size bucket?",
                "Fraud cases under FMR: number, amount, recovery rate last three years?",
                "% of originations processed under policy exceptions (LTV or KYC relaxation)."
            ],
            "financial_performance": [
                "NIM trend vs. average gold price volatility correlation (β)?",
                "Five-year risk-adjusted yield (NIM – credit cost) trajectory; inflection points?",
                "Contribution of auction gains to PBT FY24-FY25."
            ],
            "liquidity_alm": [
                "One-year cumulative ALM gap after including off-balance securitisation pay-outs?",
                "Impact on NII if CoF rises 100 bp given 52% floating-rate borrowings."
            ],
            "capital_governance": [
                "Has Tier-I CRAR ever fallen within 200 bp of regulatory minimum; what remedial actions?"
            ]
        }
    
    def _query_documents_for_question(self, question: str, cache_ids: List[str], category: str) -> Dict[str, Any]:
        if not cache_ids:
            return {"answer": "No cached documents available", "confidence": 0, "sources": []}
        
        context_prompt = f"""
        You are analyzing documents for a gold loan NBFC investment decision. 
        Category: {category}
        
        Question: {question}
        
        Analyze the available documents and provide:
        1. Direct answer with specific figures/data where available
        2. Supporting evidence from the documents
        3. Confidence level (1-5) based on data quality
        4. Any missing information needed
        5. Risk implications for investment decision
        
        If the information is not available in the documents, clearly state this.
        Focus on quantitative data and specific metrics where possible.
        """
        
        answers = []
        for cache_id in cache_ids[:3]:  # Limit to top 3 most relevant documents
            try:
                response = self._generate_response(
                    context_prompt, 
                    temperature=0.2, 
                    max_tokens=500,
                    use_cache=cache_id
                )
                answers.append({
                    "cache_id": cache_id,
                    "response": response
                })
            except Exception as e:
                self.logger.error(f"Error querying cache {cache_id}: {str(e)}")
        
        if not answers:
            return {"answer": "Unable to query documents", "confidence": 0, "sources": []}
        
        synthesis_prompt = f"""
        Synthesize these document analyses for the question: {question}
        
        Individual Responses:
        {chr(10).join([f"Source {i+1}: {ans['response']}" for i, ans in enumerate(answers)])}
        
        Provide:
        1. Consolidated answer with specific metrics
        2. Confidence level (1-5)
        3. Key supporting data points
        4. Investment implications
        
        Format as structured JSON:
        {{"answer": "...", "confidence": X, "key_metrics": [], "investment_impact": "...", "data_gaps": []}}
        """
        
        try:
            synthesis = self._generate_response(synthesis_prompt, temperature=0.3, max_tokens=400)
            result = eval(synthesis.strip())
            result["sources"] = [ans["cache_id"] for ans in answers]
            return result
        except:
            return {
                "answer": answers[0]["response"] if answers else "No response available",
                "confidence": 2,
                "sources": [ans["cache_id"] for ans in answers],
                "key_metrics": [],
                "investment_impact": "Requires further analysis",
                "data_gaps": ["Unable to synthesize responses"]
            }
    
    def _prioritize_documents_by_relevance(self, pdf_analyses: Dict, category: str) -> List[str]:
        relevance_scores = []
        
        for file_path, analysis in pdf_analyses.items():
            if "error" in analysis or "cache_id" not in analysis:
                continue
            
            content = analysis.get("content_summary", "").lower()
            classification = analysis.get("classification", {})
            usefulness = classification.get("indicative_usefulness", 0)
            
            category_keywords = {
                "business_strategy": ["business", "strategy", "branch", "digital", "products"],
                "asset_quality": ["npa", "stage", "provision", "asset quality", "restructured"],
                "underwriting_risk": ["ltv", "underwriting", "fraud", "risk", "policy"],
                "financial_performance": ["nim", "profit", "performance", "auction", "yield"],
                "liquidity_alm": ["alm", "liquidity", "maturity", "gap", "securitisation"],
                "capital_governance": ["capital", "crar", "tier", "regulatory", "governance"]
            }
            
            keyword_score = sum(1 for keyword in category_keywords.get(category, []) if keyword in content)
            total_score = usefulness * 2 + keyword_score
            
            relevance_scores.append((analysis["cache_id"], total_score, file_path))
        
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        return [cache_id for cache_id, score, path in relevance_scores[:5]]
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "qualitative_quantitative_inquiry", "Starting analyst investigation", AgentStatus.RUNNING)
        
        resource_data = process_log.get_stage_data("document_harvest")
        verification_data = process_log.get_stage_data("ingestion_qa")
        
        if not verification_data or not verification_data.get("ready_for_analysis"):
            process_log.log(self.__class__.__name__, "qualitative_quantitative_inquiry", "Verification failed", AgentStatus.FAILED)
            return {"error": "Cannot proceed without verified and complete resources"}
        
        pdf_analyses = resource_data.get("pdf_analyses", {})
        analysis_results = {
            "investigation_summary": {},
            "key_findings": [],
            "risk_flags": [],
            "data_gaps": [],
            "investment_insights": {}
        }
        
        total_questions = sum(len(questions) for questions in self.analysis_questions.values())
        processed_questions = 0
        
        for category, questions in self.analysis_questions.items():
            category_results = {}
            relevant_cache_ids = self._prioritize_documents_by_relevance(pdf_analyses, category)
            
            process_log.log(
                self.__class__.__name__, 
                f"analyzing_{category}", 
                f"Processing {len(questions)} questions", 
                AgentStatus.RUNNING
            )
            
            for question in questions:
                result = self._query_documents_for_question(question, relevant_cache_ids, category)
                category_results[question] = result
                
                if result["confidence"] >= 4:
                    analysis_results["key_findings"].append({
                        "category": category,
                        "question": question,
                        "finding": result["answer"],
                        "confidence": result["confidence"]
                    })
                
                if result["confidence"] <= 2:
                    analysis_results["risk_flags"].append({
                        "category": category,
                        "issue": "Low confidence data",
                        "question": question
                    })
                
                if result.get("data_gaps"):
                    analysis_results["data_gaps"].extend(result["data_gaps"])
                
                processed_questions += 1
                
                if processed_questions % 5 == 0:
                    process_log.log(
                        self.__class__.__name__, 
                        f"progress_update", 
                        f"{processed_questions}/{total_questions} questions processed", 
                        AgentStatus.RUNNING
                    )
            
            analysis_results["investigation_summary"][category] = category_results
        
        summary_prompt = f"""
        Summarize the key investment insights from this gold loan NBFC analysis:
        
        Categories analyzed: {len(self.analysis_questions)}
        Total findings: {len(analysis_results['key_findings'])}
        Risk flags: {len(analysis_results['risk_flags'])}
        Data gaps: {len(analysis_results['data_gaps'])}
        
        Key Findings Sample:
        {analysis_results['key_findings'][:3]}
        
        Provide:
        1. Top 3 positive investment indicators
        2. Top 3 risk concerns
        3. Critical data gaps that need follow-up
        4. Overall investment thesis readiness (1-5 scale)
        """
        
        investment_summary = self._generate_response(summary_prompt, temperature=0.3, max_tokens=600)
        analysis_results["investment_insights"]["summary"] = investment_summary
        analysis_results["investigation_summary"]["completion_rate"] = processed_questions / total_questions
        
        process_log.log(
            self.__class__.__name__, 
            "qualitative_quantitative_inquiry", 
            analysis_results, 
            AgentStatus.COMPLETED,
            f"Completed {processed_questions} questions across {len(self.analysis_questions)} categories"
        )
        
        return analysis_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/maker_agents/analyst.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    from agentic.base.base_agent import ProcessLog
    process_log = ProcessLog()
    agent = AnalystAgent()
    result = agent.execute(process_log)
    print("\n" + "="*50)
    print("ANALYST AGENT RESULTS")
    print("="*50)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Key Findings: {len(result.get('key_findings', []))}")
        print(f"Risk Flags: {len(result.get('risk_flags', []))}")
        print(f"Data Gaps: {len(result.get('data_gaps', []))}")
        print(f"Investment Insights: {result.get('investment_insights', {}).get('summary', 'N/A')}")
    print("="*50)