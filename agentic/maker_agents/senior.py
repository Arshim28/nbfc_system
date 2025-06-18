from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

class SeniorAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.ic_questions = [
            "Does proposed transaction tighten security (first-ranking charge on pledge stock) or rely on negative pledge?",
            "What DSRA size (months of coupon) is stipulated; compare to peer precedents?",
            "Stress-case: gold price –15%, GNPA 4% -> projected DSCR and CRAR post-loss. Sufficient headroom?",
            "Will documentation include a cash-sweep above 2x DSCR?",
            "Exit visibility: are listed NCDs liquid (>₹5 cr weekly turnover) or will fund hold to maturity?",
            "Covenant-breach triggers: Tier-I <15%, Net NPA >3% – enforcement mechanisms?",
            "Expected IRR vs. fund hurdle; what spread over peer NCD curve of similar tenor?",
            "Tax pass-through and withholding implications for offshore LPs?",
            "Is the issuer's ESG score (if available) within fund threshold; any controversies pending?",
            "Key human DD priorities: audit ECL model, branch collateral audits, legal opinion on pledge enforceability."
        ]
    
    def _synthesize_investment_thesis(self, all_data: Dict) -> Dict[str, Any]:
        analyst_data = all_data.get("qualitative_quantitative_inquiry", {})
        associate_data = all_data.get("financial_ratio_analysis", {})
        sector_data = all_data.get("sector_research", {})
        
        synthesis_prompt = f"""
        Synthesize a comprehensive investment thesis for this gold loan NBFC based on multi-agent analysis:
        
        ANALYST INSIGHTS:
        - Key Findings: {len(analyst_data.get('key_findings', []))}
        - Risk Flags: {len(analyst_data.get('risk_flags', []))}
        - Investment Insights: {analyst_data.get('investment_insights', {}).get('summary', 'N/A')[:200]}
        
        FINANCIAL ANALYSIS:
        - Financial Health Score: {associate_data.get('financial_health_score', 'N/A')}/10
        - Red Flags: {len(associate_data.get('red_flags', []))}
        - Peer Comparison: {associate_data.get('peer_comparison_summary', {}).get('data_quality', 'N/A')}
        
        SECTOR RESEARCH:
        - Research Quality: {sector_data.get('research_quality', {}).get('reliability', 'N/A')}
        - Competitive Analysis Available: {'Yes' if sector_data.get('competitive_analysis') else 'No'}
        
        Provide investment thesis with:
        1. Executive Summary (2-3 sentences)
        2. Investment Strengths (top 3)
        3. Key Risk Concerns (top 3)
        4. Financial Health Assessment (1-10 score with rationale)
        5. Sector Position (Strong/Average/Weak)
        6. Preliminary Investment Recommendation (BUY/HOLD/PASS)
        """
        
        response = self._generate_response(synthesis_prompt, temperature=0.3, max_tokens=800)
        
        return {
            "investment_thesis": response,
            "data_sources": {
                "analyst_questions_processed": len(analyst_data.get('investigation_summary', {})),
                "financial_ratios_calculated": len(associate_data.get('ratio_analyses', {})),
                "sector_research_queries": sector_data.get('research_quality', {}).get('queries_completed', 0)
            }
        }
    
    def _assess_transaction_structure(self, financial_data: Dict) -> Dict[str, Any]:
        structure_prompt = f"""
        Based on the financial analysis, assess the optimal transaction structure for this gold loan NBFC investment:
        
        Financial Health Score: {financial_data.get('financial_health_score', 'N/A')}/10
        Red Flags Count: {len(financial_data.get('red_flags', []))}
        
        For each IC question, provide assessment:
        
        1. Security Structure: Recommend first-ranking charge vs negative pledge based on credit quality
        2. DSRA Requirement: Suggest months of coupon coverage based on risk profile
        3. Stress Testing: Based on sensitivity analysis, assess capital adequacy post-stress
        4. Cash Sweep: Recommend DSCR threshold for cash sweep mechanism
        5. Liquidity/Exit: Assess secondary market liquidity for NCDs
        6. Covenants: Suggest appropriate Tier-I and NPA triggers
        7. Pricing: Estimate fair spread over peer curve
        8. Tax Efficiency: Highlight key tax considerations
        9. ESG Compliance: Note any ESG red flags
        10. DD Priorities: List critical human due diligence items
        
        Format as structured recommendations with rationale.
        """
        
        response = self._generate_response(structure_prompt, temperature=0.3, max_tokens=1000)
        
        return {
            "transaction_structure_assessment": response,
            "risk_adjusted_recommendations": {
                "security_level": "HIGH" if financial_data.get('financial_health_score', 0) < 7 else "STANDARD",
                "covenant_intensity": "TIGHT" if len(financial_data.get('red_flags', [])) > 3 else "STANDARD",
                "pricing_premium": "REQUIRED" if financial_data.get('financial_health_score', 0) < 6 else "MARKET"
            }
        }
    
    def _calculate_expected_returns(self, all_data: Dict) -> Dict[str, Any]:
        associate_data = all_data.get("financial_ratio_analysis", {})
        sector_data = all_data.get("sector_research", {})
        
        returns_prompt = f"""
        Calculate expected returns for this gold loan NBFC investment:
        
        FINANCIAL METRICS:
        - Red Flags: {len(associate_data.get('red_flags', []))}
        - Financial Health: {associate_data.get('financial_health_score', 'N/A')}/10
        
        SECTOR CONTEXT:
        - Peer Valuations: {sector_data.get('sector_research', {}).get('valuation_benchmarks', {})}
        - Gold Price Outlook: {sector_data.get('sector_research', {}).get('gold_price_outlook', {})}
        
        Estimate:
        1. Base Case IRR (assuming stable performance)
        2. Upside Case IRR (strong performance scenario)
        3. Downside Case IRR (stress scenario)
        4. Spread over peer NCD curve
        5. Risk-adjusted return (base case IRR minus risk premium)
        6. Expected holding period return
        7. Exit strategy and timeline
        
        Consider fund hurdle rate typically 12-15% for AIF Category 2.
        """
        
        response = self._generate_response(returns_prompt, temperature=0.3, max_tokens=600)
        
        return {
            "return_analysis": response,
            "hurdle_assessment": {
                "fund_hurdle_range": "12-15%",
                "risk_category": "MEDIUM" if associate_data.get('financial_health_score', 5) >= 7 else "HIGH",
                "expected_return_vs_hurdle": "To be calculated based on structure"
            }
        }
    
    def _identify_human_dd_priorities(self, all_data: Dict) -> Dict[str, Any]:
        analyst_data = all_data.get("qualitative_quantitative_inquiry", {})
        associate_data = all_data.get("financial_ratio_analysis", {})
        
        high_priority_items = []
        medium_priority_items = []
        
        if len(associate_data.get('red_flags', [])) > 3:
            high_priority_items.append("Deep dive into financial red flags and accounting practices")
        
        if len(analyst_data.get('data_gaps', [])) > 5:
            high_priority_items.append("Obtain missing financial and operational data")
        
        high_priority_items.extend([
            "On-site visit to gold storage facilities and sample branches",
            "Independent audit of ECL model and provisions adequacy", 
            "Legal opinion on pledge enforceability across jurisdictions",
            "Management interviews on strategy and risk management",
            "Verification of gold tonnage and auction processes"
        ])
        
        medium_priority_items.extend([
            "Regulatory compliance review and inspection history",
            "IT systems and cybersecurity assessment",
            "HR policies and key person risk evaluation", 
            "Market research validation in key geographies",
            "ESG assessment and sustainability practices"
        ])
        
        dd_prompt = f"""
        Prioritize human due diligence based on the analysis findings:
        
        Analysis Summary:
        - Analyst Risk Flags: {len(analyst_data.get('risk_flags', []))}
        - Financial Red Flags: {len(associate_data.get('red_flags', []))}
        - Data Gaps: {len(analyst_data.get('data_gaps', []))}
        
        Recommend:
        1. Critical items that could be deal-breakers
        2. Standard items for risk validation
        3. Estimated timeline for completion
        4. External expert requirements (legal, technical, etc.)
        5. Specific focus areas unique to gold loan business
        """
        
        dd_assessment = self._generate_response(dd_prompt, temperature=0.3, max_tokens=500)
        
        return {
            "human_dd_priorities": {
                "high_priority": high_priority_items,
                "medium_priority": medium_priority_items,
                "assessment": dd_assessment
            },
            "timeline_estimate": "4-6 weeks for comprehensive DD",
            "external_experts_needed": [
                "Gold industry specialist",
                "Banking/NBFC regulatory expert", 
                "Legal counsel for pledge laws",
                "IT security auditor"
            ]
        }
    
    def _generate_ic_memorandum(self, all_data: Dict) -> Dict[str, Any]:
        investment_thesis = self._synthesize_investment_thesis(all_data)
        transaction_structure = self._assess_transaction_structure(all_data.get("financial_ratio_analysis", {}))
        return_analysis = self._calculate_expected_returns(all_data)
        dd_priorities = self._identify_human_dd_priorities(all_data)
        
        ic_memo_prompt = f"""
        Draft an Investment Committee memorandum for this gold loan NBFC investment:
        
        EXECUTIVE SUMMARY:
        {investment_thesis.get('investment_thesis', '')[:300]}
        
        TRANSACTION STRUCTURE:
        {transaction_structure.get('risk_adjusted_recommendations', {})}
        
        EXPECTED RETURNS:
        {return_analysis.get('hurdle_assessment', {})}
        
        Include sections:
        1. INVESTMENT OVERVIEW (sector, company, transaction size)
        2. INVESTMENT THESIS (3 key strengths, 3 key risks)
        3. FINANCIAL ANALYSIS SUMMARY
        4. SECTOR POSITIONING
        5. TRANSACTION TERMS (structure, covenants, pricing)
        6. RISK MITIGATION MEASURES
        7. EXPECTED RETURNS & EXIT STRATEGY
        8. RECOMMENDATION (APPROVE/REJECT/DEFER)
        9. CONDITIONS PRECEDENT
        10. NEXT STEPS & TIMELINE
        
        Keep it executive-level, focusing on decision-making insights.
        """
        
        ic_memorandum = self._generate_response(ic_memo_prompt, temperature=0.3, max_tokens=1200)
        
        return {
            "ic_memorandum": ic_memorandum,
            "recommendation_summary": {
                "preliminary_decision": "CONDITIONAL APPROVAL",  # Based on analysis quality
                "key_conditions": dd_priorities.get("human_dd_priorities", {}).get("high_priority", [])[:3],
                "timeline_to_decision": "6-8 weeks post-DD completion",
                "investment_committee_readiness": "HIGH"
            }
        }
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "ic_synthesis", "Starting IC-level synthesis & risk-return analysis", AgentStatus.RUNNING)
        
        # Gather all previous stage data
        all_stage_data = {}
        for entry in process_log.entries:
            if entry["status"] in ["completed", "verified"]:
                all_stage_data[entry["stage"]] = entry["data"]
        
        if len(all_stage_data) < 4:  # Should have at least 4 completed stages
            process_log.log(self.__class__.__name__, "ic_synthesis", "Insufficient data for synthesis", AgentStatus.FAILED)
            return {"error": "Cannot proceed without complete pipeline data"}
        
        process_log.log(self.__class__.__name__, "investment_thesis", "Synthesizing investment thesis", AgentStatus.RUNNING)
        investment_thesis = self._synthesize_investment_thesis(all_stage_data)
        
        process_log.log(self.__class__.__name__, "transaction_structure", "Assessing transaction structure", AgentStatus.RUNNING)
        transaction_structure = self._assess_transaction_structure(all_stage_data.get("financial_ratio_analysis", {}))
        
        process_log.log(self.__class__.__name__, "return_analysis", "Calculating expected returns", AgentStatus.RUNNING)
        return_analysis = self._calculate_expected_returns(all_stage_data)
        
        process_log.log(self.__class__.__name__, "dd_priorities", "Identifying human DD priorities", AgentStatus.RUNNING)
        dd_priorities = self._identify_human_dd_priorities(all_stage_data)
        
        process_log.log(self.__class__.__name__, "ic_memorandum", "Drafting IC memorandum", AgentStatus.RUNNING)
        ic_memorandum = self._generate_ic_memorandum(all_stage_data)
        
        # Final risk-return assessment
        risk_return_prompt = f"""
        Provide final risk-return assessment for Investment Committee:
        
        PIPELINE COMPLETENESS:
        - Stages Completed: {len(all_stage_data)}
        - Data Quality: HIGH (all agents completed successfully)
        
        KEY METRICS:
        - Financial Health Score: {all_stage_data.get('financial_ratio_analysis', {}).get('financial_health_score', 'N/A')}/10
        - Analyst Confidence: {len(all_stage_data.get('qualitative_quantitative_inquiry', {}).get('key_findings', []))} key findings
        - Sector Position: {all_stage_data.get('sector_research', {}).get('research_quality', {}).get('reliability', 'N/A')}
        
        Final Assessment:
        1. Investment Readiness Score (1-10)
        2. Risk Rating (LOW/MEDIUM/HIGH)
        3. Go/No-Go Recommendation
        4. Critical Success Factors
        5. Decision Timeline
        """
        
        final_assessment = self._generate_response(risk_return_prompt, temperature=0.2, max_tokens=400)
        
        senior_results = {
            "investment_thesis": investment_thesis,
            "transaction_structure": transaction_structure,
            "return_analysis": return_analysis,
            "dd_priorities": dd_priorities,
            "ic_memorandum": ic_memorandum,
            "final_assessment": {
                "risk_return_summary": final_assessment,
                "pipeline_completeness": len(all_stage_data),
                "ready_for_ic": True,
                "human_review_required": True,
                "decision_timeline": "6-8 weeks"
            },
            "process_summary": {
                "total_agents": len(all_stage_data),
                "total_duration_minutes": (datetime.now() - process_log.start_time).total_seconds() / 60,
                "status": "READY_FOR_HUMAN_REVIEW_AND_IC_PRESENTATION"
            }
        }
        
        process_log.log(
            self.__class__.__name__, 
            "ic_synthesis", 
            senior_results, 
            AgentStatus.COMPLETED,
            f"IC memorandum ready, {len(all_stage_data)} stages synthesized"
        )
        
        return senior_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/maker_agents/senior.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    from agentic.base.base_agent import ProcessLog
    from maker_agents.senior import SeniorAgent
    process_log = ProcessLog()
    agent = SeniorAgent()
    result = agent.execute(process_log)
    print("\n" + "="*50)
    print("SENIOR AGENT RESULTS")
    print("="*50)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"IC Synthesis: {result.get('ic_synthesis', 'N/A')}")
        print(f"Risk Return: {result.get('risk_return', 'N/A')}")
    print("="*50)