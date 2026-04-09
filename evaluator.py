import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Any, Tuple
import logging

# Load environment variables (API Keys, etc.)
load_dotenv()

# Set up logging early for "pro" feel
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class InterviewEvaluator:
    """A professional-grade NLP evaluator for technical interview answers."""

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
        "can", "to", "of", "in", "on", "at", "by", "for", "with", "about", "as", "into", "through",
        "during", "before", "after", "above", "below", "up", "down", "out", "off", "over", "under",
        "again", "then", "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "but", "and", "or", "if", "it", "its", "this",
        "that", "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who", "also",
        "from", "between", "among", "their", "your", "our", "my", "his", "her", "any", "is", "its"
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing InterviewEvaluator with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize Gemini 2.0 (google-generativeai SDK - STABLE)
        self.gemini_enabled = False
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_key:
            logger.warning(f"CRITICAL: GEMINI_API_KEY not found in env. CWD: {os.getcwd()}")
        else:
            try:
                genai.configure(api_key=gemini_key)
                # Switch to 1.5 Flash for high stability and better rate limits
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_enabled = True
                logger.info("PRO MODE: Gemini 1.5 Flash successfully enabled.")
            except Exception as e:
                logger.error(f"Gemini Init ERROR: {e}")

    def _extract_keywords(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", text.lower())
        words = text.split()
        seen = set()
        unique_keywords = []
        for word in words:
            if word not in self.STOPWORDS and len(word) > 2 and word not in seen:
                seen.add(word)
                unique_keywords.append(word)
        return unique_keywords

    def _calculate_keyword_coverage(self, ref_keywords: List[str], user_text: str) -> Tuple[List[str], List[str], float]:
        user_lower = user_text.lower()
        matched = []
        missed = []
        for kw in ref_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', user_lower):
                matched.append(kw)
            else:
                missed.append(kw)
        coverage_pct = round((len(matched) / len(ref_keywords) * 100), 1) if ref_keywords else 0.0
        return matched, missed, coverage_pct

    async def _get_ai_critique(self, reference: str, answer: str, score: float) -> Dict[str, Any]:
        """Uses Gemini 1.5 Flash (Async) to provide a deep review in JSON format."""
        if not self.gemini_enabled:
            logger.warning("Gemini is not enabled — skipping AI critique.")
            return None
        
        prompt = f"""
        Role: Senior Technical Interviewer & Coach
        Question Reference (The Gold Standard): {reference}
        User's Actual Answer: {answer}
        Similarity Score: {score}/10
        
        Task: Provide a deep technical mentorship review in valid JSON format.
        
        Requirements:
        1. perfect_answer: A concise (~120 words) FAANG-level answer using precise terminology.
        2. explanation: A conceptual deep-dive (2-3 sentences) explaining the 'Why' behind the topic.
        3. improvement_points: 3 specific bullet points addressing gaps in the user's answer.
        4. hedging_detected: Boolean (true if the user used 'maybe', 'I think', 'I guess' etc).
        5. pro_tip: One punchy professional tips for a senior candidate.
        6. summary: A 1-sentence supportive final evaluation.
        
        Output Strictly as JSON:
        {{
            "perfect_answer": "...",
            "explanation": "...",
            "improvement_points": ["...", "...", "..."],
            "hedging_detected": false,
            "pro_tip": "...",
            "summary": "..."
        }}
        """
        try:
            logger.info(f"Invoking Gemini 1.5 Flash for answer: '{answer[:30]}...'")
            response = await self.gemini_model.generate_content_async(prompt)
            
            # Robust extraction of the text body
            if not response or not hasattr(response, 'text'):
                logger.error("Gemini response objects is invalid or blocked.")
                return None
                
            raw_text = response.text.strip()
            
            # Extract JSON from potential markdown markers (```json ... ```)
            import json
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                logger.info("Successfully parsed Gemini critique JSON.")
                return data
            
            logger.error(f"Failed to find JSON block in Gemini output: {raw_text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Gemini API failure: {str(e)}")
            return None

    def _calculate_density_penalty(self, text: str, semantic_score: float) -> Tuple[float, str]:
        """Detects if the answer is 'padded' with fluff vs actual technical content."""
        words = text.split()
        word_count = len(words)
        
        if word_count == 0: return 0.0, None
        
        # Professional technical answers are typically concise.
        # Penalty triggers if word count is very high but semantic match is low (padding).
        penalty = 0.0
        warning = None
        
        if word_count > 250 and semantic_score < 6.0:
            penalty = 1.5
            warning = "High word count with low technical density (potential padding)."
        elif word_count > 150 and semantic_score < 4.0:
            penalty = 1.0
            warning = "Answer is verbose but lacks core technical points."
            
        return penalty, warning

    async def evaluate(self, reference: str, user_answer: str) -> Dict[str, Any]:
        """Performs a multi-metric evaluation including behavior and density analysis."""
        try:
            # 1. Base NLP Metrics
            embeddings = self.model.encode([reference, user_answer])
            raw_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            BASELINE = 0.10
            normalized = max(0.0, (raw_sim - BASELINE) / (1.0 - BASELINE))
            semantic_score = round(min(normalized * 10, 10.0), 2)

            ref_keywords = self._extract_keywords(reference)
            matched_kws, missed_kws, coverage_pct = self._calculate_keyword_coverage(ref_keywords, user_answer)
            keyword_score = round(coverage_pct / 10, 2)

            # 2. Advanced Calibration (Padding & Density)
            density_penalty, density_warning = self._calculate_density_penalty(user_answer, semantic_score)

            # 3. Composite Score Calculation
            composite_score = round(max(0.0, min(0.6 * semantic_score + 0.4 * keyword_score - density_penalty, 10.0)), 2)

            # 4. Deep AI Analysis (Gaps & Confidence)
            ai_data = await self._get_ai_critique(reference, user_answer, composite_score)
            ai_data = ai_data if ai_data else {} # Safe default
            
            # 5. Behavioral Flags
            behavioral_flags = []
            if ai_data.get("hedging_detected"):
                behavioral_flags.append("Hedging language detected (lower confidence)")
            if density_warning:
                behavioral_flags.append(density_warning)

            # 6. Final Data Assembly & Post-Processing (Dynamic fallback)
            is_ai_success = bool(ai_data)
            
            # Smart Fallbacks for Explanation
            fallback_explanation = f"This topic revolves around {reference[:80]}... The core objective is Technical Precision and Semantic Alignment."
            if not is_ai_success:
                 # Extract the first sentence of the reference as a better-than-nothing explanation
                 first_sent = reference.split('.')[0] + '.'
                 fallback_explanation = f"Fundamental Concept: {first_sent} Aim for deeper coverage of the matched keywords."

            # Smart Fallbacks for Improvement Points
            fallback_improvements = ["Focus on keyword density", "Elaborate more on concepts"]
            if not is_ai_success and missed_kws:
                 fallback_improvements = [f"Incorporate the term '{kw}' in your explanation" for kw in missed_kws[:2]]
                 fallback_improvements.append("Increase technical depth by referencing the expert solution.")

            result_data = {
                "score": composite_score,
                "metrics": {
                    "semantic_similarity": semantic_score,
                    "keyword_relevance": keyword_score,
                    "coverage_percentage": coverage_pct,
                    "word_count": len(user_answer.split()),
                    "density_penalty": density_penalty
                },
                "analysis": {
                    "matched_keywords": matched_kws[:15],
                    "missed_keywords": missed_kws[:10],
                    "perfect_answer": ai_data.get("perfect_answer") if is_ai_success else reference,
                    "explanation": ai_data.get("explanation") if is_ai_success else fallback_explanation,
                    "improvement_points": ai_data.get("improvement_points") if is_ai_success else fallback_improvements,
                    "pro_tip": ai_data.get("pro_tip") if is_ai_success else "Sound more senior by using technical terminology from the reference.",
                    "feedback": ai_data.get("summary") if is_ai_success else next(msg for sc, msg in [
                        (8.5, "Strong answer with good technical coverage."),
                        (7.0, "Good answer. Consider using more precise terminology."),
                        (5.0, "Average. Missing several core concepts."),
                        (0.0, "Needs more detail. Refer to the expert solution.")
                    ] if composite_score >= sc),
                    "behavioral_flags": behavioral_flags
                },
                "status": "success"
            }
            logger.info(f"EVAL COMPLETE. Sending result with Perfect Answer: {bool(result_data['analysis']['perfect_answer'])}")
            return result_data

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return { 
                "status": "error", 
                "score": 0.0,
                "metrics": { 
                    "semantic_similarity": 0.0, 
                    "keyword_relevance": 0.0, 
                    "coverage_percentage": 0.0, 
                    "word_count": 0,
                    "density_penalty": 0.0
                },
                "analysis": { 
                    "matched_keywords": [],
                    "missed_keywords": [],
                    "perfect_answer": "",
                    "explanation": "",
                    "improvement_points": [],
                    "pro_tip": "",
                    "feedback": f"Error during evaluation: {str(e)}", 
                    "behavioral_flags": ["System error occurred"] 
                }
            }

# Singleton instance
evaluator_instance = InterviewEvaluator()

async def evaluate_answer(reference: str, user_answer: str) -> Dict[str, Any]:
    return await evaluator_instance.evaluate(reference, user_answer)