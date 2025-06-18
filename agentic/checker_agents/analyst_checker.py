from agentic.base.base_agent import BaseAgent, ProcessLog, AgentStatus
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv()

class AnalystCheckerAgent(BaseAgent):
    def __init__(self, model_id: str = "gemini-2.5-flash-lite-preview-06-17"):
        super().__init__(model_id)
        self.quality_thresholds = {
            "min_completion_rate": 0.8,
            "min_avg_confidence": 3.0,
            "max_risk_flags": 8,
            "min_key_findings": 5
        }
    
    def _evaluate_question_quality(self, investigation_summary: Dict) -> Dict[str, Any]:
        total_questions = 0
        total_confidence = 0
        high_confidence_answers = 0
        low_confidence_answers = 0
        unanswered_questions = 0
        
        category_scores = {}
        
        for category, questions_data in investigation_summary.items():
            if category == "completion_rate":
                continue
                
            category_confidence = []
            category_answered = 0
            
            for question, result in questions_data.items():
                total_questions += 1
                confidence = result.get("confidence", 0)
                total_confidence += confidence
                category_confidence.append(confidence)
                
                if confidence >= 4:
                    high_confidence_answers += 1
                    category_answered += 1
                elif confidence >= 3:
                    category_answered += 1
                elif confidence <= 1:
                    unanswered_questions += 1
                else:
                    low_confidence_answers += 1
            
            category_scores[category] = {
                "avg_confidence": sum(category_confidence) / len(category_confidence) if category_confidence else 0,
                "answered_rate": category_answered / len(questions_data),
                "question_count": len(questions_data)
            }
        
        avg_confidence = total_confidence / total_questions if total_questions > 0 else 0
        
        return {
            "total_questions": total_questions,
            "avg_confidence": avg_confidence,
            "high_confidence_rate": high_confidence_answers / total_questions if total_questions > 0 else 0,
            "low_confidence_rate": low_confidence_answers / total_questions if total_questions > 0 else 0,
            "unanswered_rate": unanswered_questions / total_questions if total_questions > 0 else 0,
            "category_scores": category_scores
        }
    
    def _validate_key_findings(self, key_findings: List[Dict]) -> Dict[str, Any]:
        if not key_findings:
            return {
                "valid": False,
                "issues": ["No key findings identified"],
                "score": 0
            }
        
        validation_prompt = f"""
        Evaluate these key findings from a gold loan NBFC investment analysis:
        
        Findings ({len(key_findings)} total):
        {chr(10).join([f"- {finding['category']}: {finding['finding'][:100]}..." for finding in key_findings[:5]])}
        
        Assessment criteria:
        1. Are findings specific and quantitative where possible?
        2. Do findings cover critical investment areas (profitability, asset quality, growth)?
        3. Are findings actionable for investment decisions?
        4. Do findings demonstrate deep understanding of gold loan business?
        5. Are findings consistent with each other?
        
        Rate each criterion (1-5) and provide overall score.
        Format: {{"scores": [X,X,X,X,X], "overall": X, "comments": "...", "valid": true/false}}
        """
        
        try:
            response = self._generate_response(validation_prompt, temperature=0.2, max_tokens=300)
            validation = eval(response.strip())
            
            validation["finding_count"] = len(key_findings)
            validation["categories_covered"] = len(set(f["category"] for f in key_findings))
            
            return validation
        except:
            return {
                "valid": False,
                "issues": ["Could not validate findings quality"],
                "score": 2,
                "finding_count": len(key_findings)
            }
    
    def _check_risk_identification(self, risk_flags: List[Dict], data_gaps: List[str]) -> Dict[str, Any]:
        risk_categories = set()
        critical_risks = []
        
        for risk in risk_flags:
            category = risk.get("category", "unknown")
            risk_categories.add(category)
            
            if any(term in risk.get("issue", "").lower() for term in ["capital", "liquidity", "fraud", "regulatory"]):
                critical_risks.append(risk)
        
        data_gap_severity = len([gap for gap in data_gaps if any(term in gap.lower() for term in ["financial", "audit", "regulatory"])])
        
        return {
            "total_risks": len(risk_flags),
            "risk_categories": list(risk_categories),
            "critical_risks": len(critical_risks),
            "data_gaps": len(data_gaps),
            "critical_data_gaps": data_gap_severity,
            "risk_coverage_adequate": len(risk_categories) >= 3
        }
    
    def _evaluate_investment_readiness(self, analysis_results: Dict) -> Dict[str, Any]:
        quality_metrics = self._evaluate_question_quality(analysis_results.get("investigation_summary", {}))
        findings_validation = self._validate_key_findings(analysis_results.get("key_findings", []))
        risk_assessment = self._check_risk_identification(
            analysis_results.get("risk_flags", []), 
            analysis_results.get("data_gaps", [])
        )
        
        readiness_score = 0
        readiness_factors = []
        
        if quality_metrics["avg_confidence"] >= self.quality_thresholds["min_avg_confidence"]:
            readiness_score += 2
            readiness_factors.append("High average confidence in answers")
        else:
            readiness_factors.append(f"Low confidence: {quality_metrics['avg_confidence']:.1f}/5.0")
        
        if len(analysis_results.get("key_findings", [])) >= self.quality_thresholds["min_key_findings"]:
            readiness_score += 2
            readiness_factors.append("Sufficient key findings identified")
        else:
            readiness_factors.append(f"Insufficient findings: {len(analysis_results.get('key_findings', []))}")
        
        if risk_assessment["risk_coverage_adequate"]:
            readiness_score += 1
            readiness_factors.append("Adequate risk coverage")
        else:
            readiness_factors.append("Limited risk identification")
        
        if findings_validation.get("valid", False):
            readiness_score += 1
            readiness_factors.append("Findings quality validated")
        else:
            readiness_factors.append("Findings quality concerns")
        
        investment_thesis_prompt = f"""
        Based on this analysis quality assessment, evaluate investment thesis readiness:
        
        Quality Metrics:
        - Average Confidence: {quality_metrics['avg_confidence']:.1f}/5.0
        - Key Findings: {len(analysis_results.get('key_findings', []))}
        - Risk Flags: {len(analysis_results.get('risk_flags', []))}
        - Data Gaps: {len(analysis_results.get('data_gaps', []))}
        
        Readiness Score: {readiness_score}/6
        
        Determine:
        1. Can we proceed to financial ratio analysis? (yes/no)
        2. Are there critical gaps that block investment decision?
        3. Confidence level in proceeding (1-5)
        4. Key action items before proceeding
        
        Respond as JSON: {{"proceed": true/false, "confidence": X, "critical_gaps": [], "action_items": []}}
        """
        
        try:
            readiness_assessment = self._generate_response(investment_thesis_prompt, temperature=0.2, max_tokens=400)
            thesis_readiness = eval(readiness_assessment.strip())
            thesis_readiness["readiness_score"] = readiness_score
            thesis_readiness["readiness_factors"] = readiness_factors
            return thesis_readiness
        except:
            return {
                "proceed": readiness_score >= 4,
                "confidence": min(readiness_score, 5),
                "critical_gaps": ["Assessment validation failed"],
                "readiness_score": readiness_score,
                "readiness_factors": readiness_factors
            }
    
    def execute(self, process_log: ProcessLog) -> Dict[str, Any]:
        process_log.log(self.__class__.__name__, "analyst_verification", "Starting analyst output verification", AgentStatus.RUNNING)
        
        analysis_data = process_log.get_stage_data("qualitative_quantitative_inquiry")
        
        if not analysis_data:
            process_log.log(self.__class__.__name__, "analyst_verification", "No analyst data found", AgentStatus.FAILED)
            return {"verified": False, "issues": ["No analyst investigation data found"]}
        
        quality_metrics = self._evaluate_question_quality(analysis_data.get("investigation_summary", {}))
        findings_validation = self._validate_key_findings(analysis_data.get("key_findings", []))
        risk_assessment = self._check_risk_identification(
            analysis_data.get("risk_flags", []), 
            analysis_data.get("data_gaps", [])
        )
        investment_readiness = self._evaluate_investment_readiness(analysis_data)
        
        verification_issues = []
        
        if quality_metrics["avg_confidence"] < self.quality_thresholds["min_avg_confidence"]:
            verification_issues.append(f"Low average confidence: {quality_metrics['avg_confidence']:.1f}")
        
        if len(analysis_data.get("key_findings", [])) < self.quality_thresholds["min_key_findings"]:
            verification_issues.append(f"Insufficient key findings: {len(analysis_data.get('key_findings', []))}")
        
        if quality_metrics["unanswered_rate"] > 0.3:
            verification_issues.append(f"High unanswered rate: {quality_metrics['unanswered_rate']:.1%}")
        
        if not findings_validation.get("valid", False):
            verification_issues.append("Key findings validation failed")
        
        overall_verification = {
            "verified": len(verification_issues) == 0 and investment_readiness.get("proceed", False),
            "issues": verification_issues,
            "quality_assessment": {
                "question_metrics": quality_metrics,
                "findings_validation": findings_validation,
                "risk_assessment": risk_assessment,
                "investment_readiness": investment_readiness
            },
            "recommendation": "PROCEED" if investment_readiness.get("proceed", False) else "REVISE",
            "confidence_score": investment_readiness.get("confidence", 0),
            "next_actions": investment_readiness.get("action_items", [])
        }
        
        status = AgentStatus.VERIFIED if overall_verification["verified"] else AgentStatus.FAILED
        process_log.log(
            self.__class__.__name__, 
            "analyst_verification", 
            overall_verification, 
            status,
            f"Recommendation: {overall_verification['recommendation']}"
        )
        
        return overall_verification