import json
from groq import Groq

class Evaluation:
    def __init__(self, corpus, model="openai/gpt-oss-120b", api_key="Your_Groq_API_Key"):
        self.corpus = corpus
        self.model = model
        self.client = Groq(api_key=api_key)
    
    def generate_queries_with_scores(self, n_queries=2, temperature=0.7):
        corpus_str = "\n".join([f"{i}: {doc}" for i, doc in enumerate(self.corpus)])
        prompt = f"""
Vous êtes un juge évaluant un ensemble de documents.
Voici le corpus de 20 documents (ID 0-19) :
{corpus_str}

Veuillez générer {n_queries} **requêtes de recherche** pertinentes pour ce corpus.
Chaque requête doit :
- être en français
- ne pas être une phrase complète mais une requête courte, comme pour un moteur de recherche
- être concise et représentative du contenu des documents

Pour chaque requête, donnez une liste de 20 entiers (0-5), où le i-ème nombre représente
la pertinence du document i pour cette requête :
- 5 : très pertinent / même contenu ou sens
- 0 : pas du tout lié

Format de sortie : JSON uniquement, comme ceci :
{{"texte de requête 1": [score0, score1, ..., score19], "texte de requête 2": [score0, score1, ..., score19], ...}}

Retournez uniquement le JSON valide, sans texte supplémentaire.
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
        json.dump(results, f, indent=2,ensure_ascii=False)
    
    print(json.dumps(results, separators=(',', ':'), ensure_ascii=False))