import sys, os, json

# Add the models folder to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

from llm_mock import LLMManager
from slm import SLM
from tools.google_tool import GoogleSearchTool
from retriever import Retriever

TOP_K_CHUNKS = 5  # Number of top chunks to pass

def main():
    retriever = Retriever()
    llm = LLMManager()
    slm = SLM(model_name="gemma:2b")
    google_tool = GoogleSearchTool()

    query = input("Enter your query: ")

    # Step 1: Retrieve relevant document chunks
    chunks = retriever.retrieve(query)[:TOP_K_CHUNKS]

    # Step 2: Run mock LLM first
    print("[ROUTER] Running Mock LLM...")
    llm_result = llm.generate(query, chunks)

    # Decide if we need SLM fallback
    if llm_result["confidence"] < 0.7 or "insufficient information" in llm_result["answer"].lower():
        print("[ROUTER] Low confidence detected. Running SLM fallback...")
        
        # Optional: augment SLM with Google search results
        search_results = google_tool.search(query)
        if search_results:
            # Convert Google search results to context-like chunks
            search_chunks = [{"text": res} for res in search_results[:TOP_K_CHUNKS]]
            chunks.extend(search_chunks)

        slm_result = slm.summarize(query, chunks)
        final_result = slm_result
        final_result["model_used"] = "SLM"
        final_result["sources"] = [c.get("source", "Document") for c in chunks]

    else:
        final_result = llm_result
        final_result["model_used"] = "LLM"
        final_result["sources"] = [c.get("source", "Document") for c in chunks]

    # Step 3: Output final structured result
    print(json.dumps(final_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
