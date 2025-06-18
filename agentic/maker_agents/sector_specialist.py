import time
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

class SectorSpecialistAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.search_tool = Tool(google_search=GoogleSearch())
        self.research_queries = [
            "Muthoot Finance Manappuram Gold Loan AUM GNPA ROA CRAR FY2024 financial results",
            "gold loan NBFC market share top 5 companies India three year trend",
            "LBMA gold price INR correlation gold loan AUM growth quarterly",
            "RBI circular November 2024 gold loan auction timeline NBFC",
            "gold loan NBFC branch productivity AUM per branch benchmark",
            "fintech gold loan companies India 500 crore AUM digital lending",
            "gold loan pledge auction rules legal PIL court cases India recent",
            "gold loan industry structural trends rural wages digital KYC recycling",
            "listed gold loan NBFC valuation P/BV P/ABV multiples peer comparison",
            "gold price forecast FY2026 demand elasticity broker consensus India"
        ]
    
    def _search_with_retry(self, query: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=f"Research and analyze: {query}. Provide specific data points, financial metrics, and recent developments.",
                    config=GenerateContentConfig(
                        tools=[self.search_tool],
                        temperature=0.3,
                        max_output_tokens=600
                    )
                )
                return response.text
            except Exception as e:
                self.logger.warning(f"Search attempt {attempt+1} failed for query: {query[:50]}... Error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return f"Search failed after {max_retries} attempts: {str(e)}"
    
    def _analyze_peer_financial_metrics(self) -> Dict[str, Any]:
        query = self.research_queries[0]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Extract specific financial metrics for Muthoot Finance and Manappuram Gold Loan from this research:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "muthoot_finance": {{
                "aum_growth_fy24": "X%",
                "gnpa_percent": "X%", 
                "average_ltv": "X%",
                "cost_of_funds": "X%",
                "roa": "X%",
                "crar": "X%"
            }},
            "manappuram": {{
                "aum_growth_fy24": "X%",
                "gnpa_percent": "X%",
                "average_ltv": "X%", 
                "cost_of_funds": "X%",
                "roa": "X%",
                "crar": "X%"
            }},
            "data_quality": "HIGH/MEDIUM/LOW"
        }}
        
        Use 'N/A' for unavailable metrics.
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.2, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not extract peer metrics", "raw_data": search_result[:500]}
    
    def _analyze_market_share_trends(self) -> Dict[str, Any]:
        query = self.research_queries[1]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Analyze the market share trends from this research:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "top_5_companies": [
                {{"name": "Company", "fy22_share": "X%", "fy23_share": "X%", "fy24_share": "X%", "trend": "GAINING/LOSING/STABLE"}}
            ],
            "market_insights": ["insight1", "insight2", "insight3"],
            "total_market_size_fy24": "₹X cr"
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=500)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze market trends", "raw_data": search_result[:500]}
    
    def _analyze_gold_price_correlation(self) -> Dict[str, Any]:
        query = self.research_queries[2]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Analyze the correlation between gold prices and AUM growth:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "correlation_coefficient": "X.XX",
            "sector_beta": "X.XX",
            "key_findings": ["finding1", "finding2"],
            "quarterly_data_points": [
                {{"quarter": "Q1FY24", "gold_price_change": "X%", "aum_growth": "X%"}}
            ]
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze correlation", "raw_data": search_result[:500]}
    
    def _analyze_regulatory_developments(self) -> Dict[str, Any]:
        query = self.research_queries[3]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Summarize RBI's November 2024 circular on gold loan auction timelines:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "circular_date": "Date",
            "key_changes": ["change1", "change2"],
            "auction_timeline_new": "X days",
            "auction_timeline_old": "X days", 
            "impact_assessment": "POSITIVE/NEGATIVE/NEUTRAL",
            "compliance_deadline": "Date",
            "industry_reaction": "Brief summary"
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.2, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze regulatory changes", "raw_data": search_result[:500]}
    
    def _benchmark_branch_productivity(self, target_company_data: Dict) -> Dict[str, Any]:
        query = self.research_queries[4]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Benchmark branch productivity metrics:
        
        Market Research: {search_result}
        
        Extract peer median branch productivity and format as JSON:
        {{
            "peer_median_aum_per_branch": "₹X cr",
            "peer_range": {{"min": "₹X cr", "max": "₹X cr"}},
            "top_performers": [
                {{"company": "Name", "aum_per_branch": "₹X cr"}}
            ],
            "industry_benchmark": "₹X cr"
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=300)
            return eval(response.strip())
        except:
            return {"error": "Could not benchmark productivity", "raw_data": search_result[:500]}
    
    def _identify_fintech_disruptors(self) -> Dict[str, Any]:
        query = self.research_queries[5]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Identify fintech companies in gold lending with >₹500 cr AUM:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "major_fintechs": [
                {{"name": "Company", "aum": "₹X cr", "business_model": "Brief", "threat_level": "HIGH/MEDIUM/LOW"}}
            ],
            "threat_assessment": "Overall threat level to traditional NBFCs",
            "key_differentiators": ["differentiator1", "differentiator2"],
            "market_disruption_timeline": "X years"
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not identify fintechs", "raw_data": search_result[:500]}
    
    def _analyze_legal_developments(self) -> Dict[str, Any]:
        query = self.research_queries[6]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Identify recent legal/PIL cases affecting gold loan pledge and auction rules:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "recent_cases": [
                {{"case_name": "Name", "court": "High Court/Supreme Court", "status": "Pending/Decided", "impact": "Brief"}}
            ],
            "rule_changes": ["change1", "change2"],
            "industry_impact": "POSITIVE/NEGATIVE/NEUTRAL",
            "compliance_requirements": ["requirement1", "requirement2"]
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze legal developments", "raw_data": search_result[:500]}
    
    def _analyze_structural_trends(self) -> Dict[str, Any]:
        query = self.research_queries[7]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Outline three key structural trends affecting gold loan industry:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "structural_trends": [
                {{
                    "trend": "Rural wage growth",
                    "description": "Brief description",
                    "impact_on_demand": "POSITIVE/NEGATIVE",
                    "timeline": "Short/Medium/Long term"
                }},
                {{
                    "trend": "Digital KYC adoption", 
                    "description": "Brief description",
                    "impact_on_operations": "POSITIVE/NEGATIVE",
                    "timeline": "Short/Medium/Long term"
                }},
                {{
                    "trend": "Gold recycling patterns",
                    "description": "Brief description", 
                    "impact_on_supply": "POSITIVE/NEGATIVE",
                    "timeline": "Short/Medium/Long term"
                }}
            ]
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=500)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze trends", "raw_data": search_result[:500]}
    
    def _analyze_valuation_benchmarks(self) -> Dict[str, Any]:
        query = self.research_queries[8]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Extract valuation multiples for listed gold loan NBFCs:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "peer_multiples": [
                {{"company": "Name", "p_bv": "X.X", "p_abv": "X.X", "market_cap": "₹X cr"}}
            ],
            "average_p_bv": "X.X",
            "average_p_abv": "X.X", 
            "valuation_range": {{"p_bv_min": "X.X", "p_bv_max": "X.X"}},
            "premium_discount_factors": ["factor1", "factor2"]
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.2, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze valuations", "raw_data": search_result[:500]}
    
    def _analyze_gold_price_outlook(self) -> Dict[str, Any]:
        query = self.research_queries[9]
        search_result = self._search_with_retry(query)
        
        analysis_prompt = f"""
        Summarize broker consensus on FY26 gold price outlook:
        
        {search_result}
        
        Extract and format as JSON:
        {{
            "fy26_price_target": "₹X per 10g",
            "current_price": "₹X per 10g",
            "expected_change": "X%",
            "demand_elasticity": "High/Medium/Low",
            "key_drivers": ["driver1", "driver2"],
            "risk_factors": ["risk1", "risk2"],
            "broker_consensus": "BULLISH/BEARISH/NEUTRAL"
        }}
        """
        
        try:
            response = self._generate_response(analysis_prompt, temperature=0.3, max_tokens=400)
            return eval(response.strip())
        except:
            return {"error": "Could not analyze price outlook", "raw_data": search_result[:500]}
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "sector_research", "Starting external benchmark & macro analysis", AgentStatus.RUNNING)
        
        associate_data = process_log.get_stage_data("financial_ratio_analysis")
        if not associate_data:
            process_log.log(self.__class__.__name__, "sector_research", "No associate data found", AgentStatus.FAILED)
            return {"error": "Cannot proceed without associate financial analysis"}
        
        sector_research = {}
        
        process_log.log(self.__class__.__name__, "peer_metrics", "Analyzing peer financial metrics", AgentStatus.RUNNING)
        sector_research["peer_financial_metrics"] = self._analyze_peer_financial_metrics()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "market_share", "Analyzing market share trends", AgentStatus.RUNNING)
        sector_research["market_share_trends"] = self._analyze_market_share_trends()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "gold_correlation", "Analyzing gold price correlation", AgentStatus.RUNNING)
        sector_research["gold_price_correlation"] = self._analyze_gold_price_correlation()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "regulatory", "Analyzing regulatory developments", AgentStatus.RUNNING)
        sector_research["regulatory_developments"] = self._analyze_regulatory_developments()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "productivity", "Benchmarking branch productivity", AgentStatus.RUNNING)
        sector_research["branch_productivity"] = self._benchmark_branch_productivity({})
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "fintechs", "Identifying fintech disruptors", AgentStatus.RUNNING)
        sector_research["fintech_disruptors"] = self._identify_fintech_disruptors()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "legal", "Analyzing legal developments", AgentStatus.RUNNING)
        sector_research["legal_developments"] = self._analyze_legal_developments()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "trends", "Analyzing structural trends", AgentStatus.RUNNING)
        sector_research["structural_trends"] = self._analyze_structural_trends()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "valuations", "Analyzing valuation benchmarks", AgentStatus.RUNNING)
        sector_research["valuation_benchmarks"] = self._analyze_valuation_benchmarks()
        time.sleep(2)
        
        process_log.log(self.__class__.__name__, "outlook", "Analyzing gold price outlook", AgentStatus.RUNNING)
        sector_research["gold_price_outlook"] = self._analyze_gold_price_outlook()
        
        competitive_summary = f"""
        Based on the sector research, provide a competitive positioning summary:
        
        Peer Performance: {sector_research.get('peer_financial_metrics', {})}
        Market Trends: {sector_research.get('market_share_trends', {})}
        Regulatory Impact: {sector_research.get('regulatory_developments', {})}
        
        Summarize:
        1. Competitive strengths vs peers
        2. Market position assessment  
        3. Key sector headwinds/tailwinds
        4. Investment attractiveness in current context
        """
        
        competitive_analysis = self._generate_response(competitive_summary, temperature=0.3, max_tokens=600)
        
        results = {
            "sector_research": sector_research,
            "competitive_analysis": competitive_analysis,
            "research_quality": {
                "queries_completed": len([r for r in sector_research.values() if "error" not in str(r)]),
                "total_queries": len(self.research_queries),
                "data_freshness": "CURRENT",
                "reliability": "HIGH" if len([r for r in sector_research.values() if "error" not in str(r)]) >= 8 else "MEDIUM"
            }
        }
        
        process_log.log(
            self.__class__.__name__, 
            "sector_research", 
            results, 
            AgentStatus.COMPLETED,
            f"Completed {results['research_quality']['queries_completed']}/{len(self.research_queries)} research queries"
        )
        
        return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/maker_agents/sector_specialist.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    from agentic.base.base_agent import ProcessLog
    from maker_agents.sector_specialist import SectorSpecialistAgent
    process_log = ProcessLog()
    agent = SectorSpecialistAgent()
    result = agent.execute(process_log)
    print("\n" + "="*50)
    print("SECTOR SPECIALIST AGENT RESULTS")
    print("="*50)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Sector Insights: {result.get('sector_insights', 'N/A')}")
        print(f"Benchmarks: {result.get('benchmarks', 'N/A')}")
    print("="*50)