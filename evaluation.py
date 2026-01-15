import json
from groq import Groq

class Evaluation:
    def __init__(self, corpus, model="openai/gpt-oss-120b", api_key="YOUR_GROQ_API_KEY"):
        self.corpus = corpus
        self.model = model
        self.client = Groq(api_key=api_key)
    
    def generate_queries_with_scores(self, n_queries=2, temperature=0.7):
        corpus_str = "\n".join([f"{i}: {doc}" for i, doc in enumerate(self.corpus)])
        prompt = f"""
You are a judge evaluating a set of documents.
Here is the corpus of 20 documents (ID 0-19):
{corpus_str}

Please generate {n_queries} queries relevant to this corpus.
For each query, give a list of 20 integers (0-5), where the i-th number represents
how relevant the document i is to this query:
- 5: very relevant / same meaning
- 0: not related at all

Output format as JSON: 
{{"query text 1": [score0, score1, ..., score19], ...}}

Only return valid JSON, no other text.
"""
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=8192,
            top_p=1,
            stream=False  # Set to False to get complete response at once
        )
        
        content = completion.choices[0].message.content
        
        # Clean up the response in case there's markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)

if __name__ == "__main__":
    from corpus import CORPUS
    evaluator = Evaluation(CORPUS)
    results = evaluator.generate_queries_with_scores(10)
    
    # Save to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(json.dumps(results, separators=(',', ':')))
