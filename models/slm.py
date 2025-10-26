import sys, os, re, subprocess, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


class SLM:
    """
    Small Language Model using local Gemma 2B via Ollama.
    Two-phase adaptive reasoning:
    1) Direct query
    2) Context-assisted fallback
    Guard rails enforced, concise and factual output.
    """

    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.forbidden_patterns = [
            r"password", r"private key", r"credit card number", r"ssn"
        ]

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def guard_rails(self, text: str) -> bool:
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return False
        return True

    def run_gemma(self, prompt: str) -> str:
        """Run prompt through local Gemma 2B using Ollama CLI."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=90
            )
            output = result.stdout.strip()
            return output if output else "Insufficient information provided in context."
        except Exception as e:
            return f"[SLM ERROR] {e}"

    def summarize(self, query: str, context_chunks: list) -> dict:
        """Two-phase reasoning: direct then context-assisted fallback."""
        if not context_chunks:
            return {"answer": "No context retrieved.", "confidence": 0.0, "status": "fallback"}

        # Phase 1: Direct reasoning with query only
        direct_prompt = f"You are a PCI DSS expert. Answer briefly:\n\nQuery: {query}"
        direct_answer = self.run_gemma(direct_prompt)

        # Quick quality check
        if len(direct_answer) > 50 and any(k.lower() in direct_answer.lower() for k in query.split()):
            return {
                "answer": direct_answer.strip(),
                "confidence": 0.9,
                "status": "direct"
            }

        # Phase 2: Context-assisted reasoning
        context = "\n\n".join(
            self.clean_text(c["text"]) for c in context_chunks[:3]
            if self.guard_rails(c["text"])
        )

        refined_prompt = f"""
You are a PCI DSS expert. Answer the query using only the following context.
Do NOT hallucinate. Use factual content only.

Format:
- 5-line summary
- 3–5 short bullet points (max 12 words each)
- Clear, compact, professional tone

Context:
{context}

Query:
{query}
        """.strip()

        refined_answer = self.run_gemma(refined_prompt)

        confidence = min(0.95, 0.6 + 0.1 * np.log1p(len(context_chunks)))

        return {
            "answer": refined_answer.strip(),
            "confidence": round(float(confidence), 2),
            "status": "context_refined"
        }


# ===== TEST SCRIPT =====
if __name__ == "__main__":
    from retriever import Retriever

    retriever = Retriever()
    query = input("Enter a question: ")
    chunks = retriever.retrieve(query)

    slm = SLM(model_name="gemma:2b")
    response = slm.summarize(query, chunks)
    print(json.dumps(response, indent=2, ensure_ascii=False))
