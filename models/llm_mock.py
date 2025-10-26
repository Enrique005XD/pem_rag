import sys, os, json, random, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import google.generativeai as genai
except ImportError:
    genai = None

SEARCH_RESULTS_FILE = "tools/search_results.json"


class LLMManager:
    """
    Pipeline:
    1. Mock LLM generates preliminary context-based answer.
    2. Gemini refines with guardrails for factual, readable output.
    3. Fallback: SLM (mock or small model) if Gemini fails.
    """

    def __init__(self):
        self.api_key = "AIzaSyAte3m51n9cxuHtJWuASyj4SgJcgTksLXU"
        self.gemini_ready = False

        if genai and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                self.gemini_ready = True
                print("[INIT] Gemini API client initialized.")
            except Exception as e:
                print(f"[INIT ERROR] Gemini init failed: {e}")
        else:
            print("[INIT] Running in mock mode only.")

    # ===== MOCK SECTION =====
    def _mock_generate(self, query, context_chunks):
        context = " ".join([c["text"][:300] for c in context_chunks[:2]])
        answer = f"Preliminary summary based on PCI DSS documents:\n{context[:600]}..."
        confidence = round(random.uniform(0.6, 0.9), 2)
        return {
            "answer": answer,
            "confidence": confidence,
            "status": "success",
            "search_results": []
        }

    # ===== GEMINI REFINEMENT =====
    def _gemini_refine(self, query, mock_answer, context_chunks):
        if not self.gemini_ready:
            print("[Gemini] Not initialized, skipping.")
            return mock_answer

        try:
            context = "\n\n".join([c["text"] for c in context_chunks[:3]])
            prompt = f"""
You are an expert PCI DSS analyst.
Answer using only the provided context. Do not hallucinate.
Respond in **clean, readable English** with:
- A one-line summary
- Bulleted core points
- Avoid repeating sentences
- If info is missing, say: "Insufficient information provided in context."

Context:
{context}

Mock draft:
{mock_answer['answer']}

Question: {query}
"""

            response = self.model.generate_content(prompt)
            refined_text = self._format_output(response.text)
            return {
                "answer": refined_text,
                "confidence": 0.95,
                "status": "success",
                "search_results": mock_answer["search_results"]
            }

        except Exception as e:
            print(f"[Gemini ERROR] {e}")
            return {
                "answer": self._format_output(mock_answer["answer"]),
                "confidence": mock_answer["confidence"],
                "status": "fallback",
                "search_results": mock_answer["search_results"]
            }

    # ===== OUTPUT FORMATTING =====
    def _format_output(self, text):
        # Remove markdown characters and excessive newlines
        text = re.sub(r'[*_#]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        text = text.replace("•", "-")

        # Deduplicate sentences (simple method)
        lines = text.split("\n")
        seen = set()
        clean_lines = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                clean_lines.append(line_clean)
                seen.add(line_clean)

        # Add "Insufficient information" note if last line is vague
        last_line = clean_lines[-1] if clean_lines else ""
        if any(w in last_line.lower() for w in ["does not specify", "insufficient"]):
            clean_lines[-1] = "Insufficient information provided in context."

        return "\n\n".join(clean_lines)

    # ===== PIPELINE =====
    def generate(self, query, context_chunks):
        print("[LLM] Step 1: Running Mock LLM...")
        mock_result = self._mock_generate(query, context_chunks)

        print("[LLM] Step 2: Refining with Gemini...")
        final_result = self._gemini_refine(query, mock_result, context_chunks)
        return final_result


# ===== TEST =====
if __name__ == "__main__":
    from retriever import Retriever
    retriever = Retriever()
    query = input("Enter a question: ")
    chunks = retriever.retrieve(query)
    llm = LLMManager()
    response = llm.generate(query, chunks)
    print(response["answer"])
