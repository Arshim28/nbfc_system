import pandas as pd
import numpy as np
from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

class AssociateAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.peer_benchmarks = {
            "muthoot_manappuram": {
                "debt_to_aum": {"median": 0.65, "q1": 0.58, "q3": 0.72},
                "gnpa_percent": {"median": 2.1, "q1": 1.8, "q3": 2.5},
                "cost_to_income": {"median": 0.42, "q1": 0.38, "q3": 0.46},
                "roa": {"median": 3.2, "q1": 2.8, "q3": 3.6},
                "nim": {"median": 11.5, "q1": 10.8, "q3": 12.2}
            }
        }
        
        self.ratio_calculations = [
            "debt_to_aum",
            "gnpa_consistency_check", 
            "stage3_coverage_ratio",
            "interest_coverage",
            "roa_decomposition",
            "cost_to_income",
            "liquidity_coverage",
            "sensitivity_analysis",
            "accounting_red_flags",
            "peer_comparison"
        ]
    
    def _extract_financial_data_from_csvs(self, csv_analyses: Dict) -> Dict[str, Any]:
        financial_data = {
            "balance_sheet": {},
            "profit_loss": {},
            "cash_flow": {},
            "asset_quality": {},
            "alm_data": {}
        }
        
        for csv_path, analysis in csv_analyses.items():
            try:
                df = pd.read_csv(csv_path)
                data_type = analysis.get("data_type", "").lower()
                
                extract_prompt = f"""
                Extract key financial metrics from this {data_type} data for gold loan NBFC ratio analysis:
                
                Data shape: {df.shape}
                Columns: {df.columns.tolist()}
                Sample data:
                {df.head(3).to_string()}
                
                Extract and format as JSON (use null for missing values):
                {{
                    "total_assets": value,
                    "total_aum": value,
                    "total_debt": value,
                    "tier1_capital": value,
                    "net_interest_income": value,
                    "operating_expenses": value,
                    "profit_before_tax": value,
                    "interest_expense": value,
                    "gross_npa": value,
                    "net_npa": value,
                    "provisions": value,
                    "stage3_assets": value,
                    "cash_and_equivalents": value,
                    "fiscal_year": "FYXXXX"
                }}
                
                Only extract values that are clearly identifiable in the data.
                """
                
                response = self._generate_response(extract_prompt, temperature=0.1, max_tokens=400)
                try:
                    extracted_data = eval(response.strip())
                    fy = extracted_data.get("fiscal_year", "unknown")
                    
                    if "balance" in data_type or "bs" in data_type:
                        financial_data["balance_sheet"][fy] = extracted_data
                    elif "profit" in data_type or "p&l" in data_type:
                        financial_data["profit_loss"][fy] = extracted_data
                    elif "cash" in data_type:
                        financial_data["cash_flow"][fy] = extracted_data
                    elif "asset quality" in data_type or "npa" in data_type:
                        financial_data["asset_quality"][fy] = extracted_data
                    elif "alm" in data_type:
                        financial_data["alm_data"][fy] = extracted_data
                        
                except:
                    self.logger.warning(f"Could not parse financial data from {csv_path}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {csv_path}: {str(e)}")
        
        return financial_data
    
    def _calculate_debt_to_aum_ratio(self, financial_data: Dict) -> Dict[str, Any]:
        ratios = {}
        
        for fy in financial_data.get("balance_sheet", {}):
            bs_data = financial_data["balance_sheet"][fy]
            total_debt = bs_data.get("total_debt")
            total_aum = bs_data.get("total_aum")
            
            if total_debt and total_aum:
                ratio = total_debt / total_aum
                peer_median = self.peer_benchmarks["muthoot_manappuram"]["debt_to_aum"]["median"]
                
                ratios[fy] = {
                    "debt_to_aum": ratio,
                    "peer_median": peer_median,
                    "delta_vs_peer": ratio - peer_median,
                    "flag": "RED" if abs(ratio - peer_median) > 0.15 else "GREEN"
                }
        
        return {
            "calculation": "Debt/AUM Ratio Analysis",
            "ratios_by_year": ratios,
            "trend": "IMPROVING" if len(ratios) > 1 and list(ratios.values())[-1]["debt_to_aum"] < list(ratios.values())[0]["debt_to_aum"] else "STABLE"
        }
    
    def _verify_gnpa_consistency(self, financial_data: Dict) -> Dict[str, Any]:
        consistency_check = {}
        
        for fy in set(financial_data.get("profit_loss", {}).keys()) & set(financial_data.get("asset_quality", {}).keys()):
            pl_data = financial_data["profit_loss"][fy]
            aq_data = financial_data["asset_quality"][fy]
            
            pl_gnpa = pl_data.get("gross_npa")
            aq_gnpa = aq_data.get("gross_npa")
            
            if pl_gnpa and aq_gnpa:
                difference = abs(pl_gnpa - aq_gnpa)
                consistency_check[fy] = {
                    "pl_gnpa": pl_gnpa,
                    "aq_gnpa": aq_gnpa,
                    "difference": difference,
                    "consistent": difference < 0.01,
                    "flag": "GREEN" if difference < 0.01 else "RED"
                }
        
        return {
            "calculation": "GNPA Consistency Check",
            "checks_by_year": consistency_check,
            "overall_consistent": all(check["consistent"] for check in consistency_check.values())
        }
    
    def _calculate_stage3_coverage(self, financial_data: Dict) -> Dict[str, Any]:
        coverage_ratios = {}
        
        for fy in financial_data.get("asset_quality", {}):
            aq_data = financial_data["asset_quality"][fy]
            stage3_assets = aq_data.get("stage3_assets")
            provisions = aq_data.get("provisions")
            
            if stage3_assets and provisions:
                coverage_ratio = provisions / stage3_assets
                coverage_ratios[fy] = {
                    "stage3_assets": stage3_assets,
                    "provisions": provisions,
                    "coverage_ratio": coverage_ratio,
                    "flag": "RED" if coverage_ratio < 0.5 else "GREEN"
                }
        
        return {
            "calculation": "Stage-3 Coverage Ratio",
            "ratios_by_year": coverage_ratios,
            "adequate_coverage": all(ratio["coverage_ratio"] >= 0.5 for ratio in coverage_ratios.values())
        }
    
    def _calculate_interest_coverage(self, financial_data: Dict) -> Dict[str, Any]:
        coverage_ratios = {}
        
        for fy in financial_data.get("profit_loss", {}):
            pl_data = financial_data["profit_loss"][fy]
            pbt = pl_data.get("profit_before_tax")
            interest_expense = pl_data.get("interest_expense")
            
            if pbt and interest_expense and interest_expense > 0:
                coverage = (pbt + interest_expense) / interest_expense
                coverage_ratios[fy] = {
                    "pbt": pbt,
                    "interest_expense": interest_expense,
                    "coverage_ratio": coverage,
                    "flag": "RED" if coverage < 1.5 else "GREEN"
                }
        
        trend = "STABLE"
        if len(coverage_ratios) >= 2:
            values = [ratio["coverage_ratio"] for ratio in coverage_ratios.values()]
            trend = "IMPROVING" if values[-1] > values[0] else "DECLINING"
        
        return {
            "calculation": "Interest Coverage (PBT + Interest) / Interest",
            "ratios_by_year": coverage_ratios,
            "trend_fy21_fy25": trend
        }
    
    def _decompose_roa(self, financial_data: Dict) -> Dict[str, Any]:
        roa_components = {}
        
        for fy in financial_data.get("profit_loss", {}):
            if fy in financial_data.get("balance_sheet", {}):
                pl_data = financial_data["profit_loss"][fy]
                bs_data = financial_data["balance_sheet"][fy]
                
                nii = pl_data.get("net_interest_income", 0)
                opex = pl_data.get("operating_expenses", 0)
                pbt = pl_data.get("profit_before_tax", 0)
                total_assets = bs_data.get("total_assets", 1)
                
                if total_assets:
                    nim = nii / total_assets
                    opex_ratio = opex / total_assets
                    roa = pbt / total_assets
                    
                    roa_components[fy] = {
                        "nim": nim,
                        "opex_ratio": opex_ratio,
                        "credit_cost": max(0, nim - opex_ratio - roa),
                        "roa": roa,
                        "largest_driver": max([("NIM", nim), ("OpEx", -opex_ratio), ("Credit", -(nim - opex_ratio - roa))], key=lambda x: abs(x[1]))[0]
                    }
        
        return {
            "calculation": "ROA Decomposition (NIM - OpEx - Credit Cost)",
            "components_by_year": roa_components,
            "fy24_fy25_driver": roa_components.get("FY2025", {}).get("largest_driver", "Unknown")
        }
    
    def _calculate_cost_to_income(self, financial_data: Dict) -> Dict[str, Any]:
        cost_ratios = {}
        
        for fy in financial_data.get("profit_loss", {}):
            pl_data = financial_data["profit_loss"][fy]
            opex = pl_data.get("operating_expenses")
            nii = pl_data.get("net_interest_income", 0)
            
            net_revenue = nii
            if opex and net_revenue and net_revenue > 0:
                cost_ratio = opex / net_revenue
                peer_q1 = self.peer_benchmarks["muthoot_manappuram"]["cost_to_income"]["q1"]
                peer_q3 = self.peer_benchmarks["muthoot_manappuram"]["cost_to_income"]["q3"]
                
                flag = "GREEN"
                if cost_ratio < peer_q1 or cost_ratio > peer_q3:
                    flag = "RED"
                
                cost_ratios[fy] = {
                    "opex": opex,
                    "net_revenue": net_revenue,
                    "cost_to_income": cost_ratio,
                    "peer_25th": peer_q1,
                    "peer_75th": peer_q3,
                    "flag": flag
                }
        
        return {
            "calculation": "Cost-to-Income (OpEx / Net Revenue)",
            "ratios_by_year": cost_ratios,
            "peer_comparison": "Within peer band" if all(r["flag"] == "GREEN" for r in cost_ratios.values()) else "Outlier"
        }
    
    def _perform_sensitivity_analysis(self, financial_data: Dict) -> Dict[str, Any]:
        base_case = {}
        stress_case = {}
        
        latest_fy = max(financial_data.get("balance_sheet", {}).keys(), default="FY2025")
        if latest_fy in financial_data.get("balance_sheet", {}):
            bs_data = financial_data["balance_sheet"][latest_fy]
            
            tier1_capital = bs_data.get("tier1_capital", 0)
            total_assets = bs_data.get("total_assets", 1)
            stage3_current = bs_data.get("stage3_assets", 0)
            
            base_cet1 = tier1_capital / total_assets if total_assets else 0
            
            gold_price_impact = total_assets * 0.15 * 0.3
            stage3_stress = total_assets * 0.04
            additional_provisions = (stage3_stress - stage3_current) * 0.5
            
            stress_capital = tier1_capital - gold_price_impact - additional_provisions
            stress_cet1 = stress_capital / total_assets if total_assets else 0
            
            base_case = {
                "tier1_capital": tier1_capital,
                "cet1_ratio": base_cet1,
                "stage3_percent": stage3_current / total_assets if total_assets else 0
            }
            
            stress_case = {
                "assumptions": "Gold price -15%, Stage-3 to 4%",
                "tier1_capital": stress_capital,
                "cet1_ratio": stress_cet1,
                "capital_impact": gold_price_impact + additional_provisions,
                "sufficient_headroom": stress_cet1 > 0.12
            }
        
        return {
            "calculation": "Sensitivity Analysis",
            "base_case": base_case,
            "stress_case": stress_case,
            "headroom_adequate": stress_case.get("sufficient_headroom", False)
        }
    
    def _identify_accounting_red_flags(self, financial_data: Dict) -> Dict[str, Any]:
        red_flags = []
        
        for fy in financial_data.get("profit_loss", {}):
            pl_data = financial_data["profit_loss"][fy]
            
            other_income = pl_data.get("other_operating_income", 0)
            nii = pl_data.get("net_interest_income", 1)
            
            if other_income and nii and (other_income / nii) > 0.2:
                red_flags.append({
                    "year": fy,
                    "flag": "High Other Operating Income",
                    "ratio": other_income / nii,
                    "concern": "Unusually high non-core income"
                })
        
        return {
            "calculation": "Accounting Red Flags Analysis",
            "red_flags": red_flags,
            "clean_accounts": len(red_flags) == 0
        }
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "financial_ratio_analysis", "Starting financial ratio deep-dive", AgentStatus.RUNNING)
        
        resource_data = process_log.get_stage_data("document_harvest")
        analyst_verification = process_log.get_stage_data("analyst_verification")
        
        if not analyst_verification or not analyst_verification.get("verified"):
            process_log.log(self.__class__.__name__, "financial_ratio_analysis", "Analyst verification failed", AgentStatus.FAILED)
            return {"error": "Cannot proceed without verified analyst output"}
        
        csv_analyses = resource_data.get("csv_analyses", {})
        financial_data = self._extract_financial_data_from_csvs(csv_analyses)
        
        ratio_analyses = {}
        
        ratio_analyses["debt_to_aum"] = self._calculate_debt_to_aum_ratio(financial_data)
        ratio_analyses["gnpa_consistency"] = self._verify_gnpa_consistency(financial_data)
        ratio_analyses["stage3_coverage"] = self._calculate_stage3_coverage(financial_data)
        ratio_analyses["interest_coverage"] = self._calculate_interest_coverage(financial_data)
        ratio_analyses["roa_decomposition"] = self._decompose_roa(financial_data)
        ratio_analyses["cost_to_income"] = self._calculate_cost_to_income(financial_data)
        ratio_analyses["sensitivity_analysis"] = self._perform_sensitivity_analysis(financial_data)
        ratio_analyses["accounting_red_flags"] = self._identify_accounting_red_flags(financial_data)
        
        ratio_table = []
        red_flags = []
        
        for analysis_name, analysis_data in ratio_analyses.items():
            if "ratios_by_year" in analysis_data:
                for fy, ratio_data in analysis_data["ratios_by_year"].items():
                    for metric, value in ratio_data.items():
                        if isinstance(value, (int, float)):
                            flag = ratio_data.get("flag", "GREEN")
                            ratio_table.append({
                                "metric": f"{analysis_name}_{metric}",
                                "fiscal_year": fy,
                                "value": value,
                                "flag": flag
                            })
                            
                            if flag == "RED":
                                red_flags.append({
                                    "metric": f"{analysis_name}_{metric}",
                                    "year": fy,
                                    "value": value,
                                    "concern": "Outside peer band or threshold"
                                })
        
        results = {
            "ratio_analyses": ratio_analyses,
            "ratio_table": ratio_table,
            "red_flags": red_flags,
            "financial_health_score": max(0, 10 - len(red_flags)),
            "peer_comparison_summary": {
                "metrics_analyzed": len(ratio_analyses),
                "red_flags_count": len(red_flags),
                "data_quality": "HIGH" if len(ratio_table) > 20 else "MEDIUM"
            }
        }
        
        process_log.log(
            self.__class__.__name__, 
            "financial_ratio_analysis", 
            results, 
            AgentStatus.COMPLETED,
            f"Analyzed {len(ratio_analyses)} ratio categories, {len(red_flags)} red flags"
        )
        
        return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run agentic/maker_agents/associate.py <data_directory>")
        sys.exit(1)
    data_dir = sys.argv[1]
    from agentic.base.base_agent import ProcessLog
    from maker_agents.associate import AssociateAgent
    process_log = ProcessLog()
    agent = AssociateAgent()
    result = agent.execute(process_log)
    print("\n" + "="*50)
    print("ASSOCIATE AGENT RESULTS")
    print("="*50)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Key Ratios: {result.get('key_ratios', 'N/A')}")
        print(f"Findings: {result.get('findings', 'N/A')}")
    print("="*50)