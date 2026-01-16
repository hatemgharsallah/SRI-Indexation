import json
import os
from groq import Groq
import pandas as pd
from model_language import LanguageModel
from retrieval_model import RetrievalModel
from corpus import Corpus, Query
class Evaluation:
    def __init__(self, corpus, model="openai/gpt-oss-120b", api_key="hello"):
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
        
        results = {}

        # ⚠️ ton code existant ici (LLM judge, etc.)
        # Exemple :
        for query, scores in json.loads(content).items():
            key = str(Query(query))
            results[key] = scores  # scores = list[int] de taille 20

        return results
         
    
    def get_queries_with_scores(
        self,
        n_queries: int = 2,
        cache_file: str = "evaluation_queries_with_scores.json"
    ) -> dict:
        """
        Load queries with scores from cache if possible, otherwise generate and save to cache.
        """

        # Case 1: cache exists
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            if len(cached_data) >= n_queries:
                # Return only first n queries (order preserved)
                return dict(list(cached_data.items())[:n_queries])

        # Case 2: cache missing or insufficient
        print("⚠️ Cache missing or insufficient → generating queries...")

        generated = self.generate_queries_with_scores(n_queries)

        # Save / overwrite cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)

        return generated

    def get_queries_with_important_docs(self, n_queries=2, threshold=3):
        """
        Calls evaluator.get_queries_with_scores and returns
        a dict mapping each query to the list of relevant document IDs.
        
        Args:
            evaluator: instance of Evaluation
            n_queries (int): number of queries to generate
            threshold (int): minimum relevance score to keep a document
        
        Returns:
            dict: { query: [doc_id, doc_id, ...] }
        """
        queries_with_scores = self.get_queries_with_scores(n_queries)

        result = {}
        for query, scores in queries_with_scores.items():
            important_docs = [
                doc_id for doc_id, score in enumerate(scores)
                if score >= threshold
            ]
            result[query] = important_docs

        return result
    
    def evaluate_no_rank_metric(
        self,
        model: RetrievalModel,
        n_queries: int = 10,
        threshold: int = 3,
        limit: int = 4
    ) -> pd.DataFrame:
        """
        Evaluate a retrieval model using Precision and Recall (no ranking metrics).

        Args:
            model (RetrievalModel): retrieval model instance
            n_queries (int): number of queries to evaluate
            threshold (int): relevance threshold (>= threshold = relevant)

        Returns:
            pd.DataFrame with columns:
            ['query', 'precision', 'recall']
        """

        # Ground truth: relevant docs per query
        gt_queries = self.get_queries_with_important_docs(
            n_queries=n_queries,
            threshold=threshold
        )

        rows = []

        for query_text, relevant_docs in gt_queries.items():
            relevant_docs = set(relevant_docs)

            # Model prediction
            ranked_docs = model.rank(Query(query_text), limit=limit)
            predicted_docs = {doc_id for doc_id, _ in ranked_docs}

            # True Positives
            true_positives = predicted_docs.intersection(relevant_docs)

            # Metrics
            recall = (
                len(true_positives) / len(relevant_docs)
                if len(relevant_docs) > 0 else 0.0
            )

            precision = (
                len(true_positives) / len(predicted_docs)
                if len(predicted_docs) > 0 else 0.0
            )

            rows.append({
                "query": query_text,
                "precision": precision,
                "recall": recall
            })

        return pd.DataFrame(rows)


    def evaluate_rank_metric(
        self,
        model: RetrievalModel,
        n_queries: int = 2,
        threshold: int = 3,
        limit: int = 4
    ) -> pd.DataFrame:
        """
        Evaluate a retrieval model by rank: compute cumulative precision and recall
        at each rank for each query.

        Args:
            model: RetrievalModel instance
            n_queries: number of queries to evaluate
            threshold: relevance threshold for ground truth
            limit: max number of documents returned by rank()

        Returns:
            pd.DataFrame with columns: ['query', 'rank', 'precision', 'recall', 'doc_id']
        """

        # Ground truth: relevant docs per query
        gt_queries = self.get_queries_with_important_docs(n_queries=n_queries, threshold=threshold)

        rows = []

        for query_text, relevant_docs in gt_queries.items():
            relevant_docs = set(relevant_docs)
            query_obj = Query(query_text)
            ranked_docs = model.rank(query_obj, limit=limit)

            num_relevant_found = 0

            for i, (doc_id, score) in enumerate(ranked_docs, start=1):  # i = rank (1-indexed)
                if doc_id in relevant_docs:
                    num_relevant_found += 1

                recall = num_relevant_found / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
                precision = num_relevant_found / i

                rows.append({
                    "query": query_text,
                    "rank": i,
                    "doc_id": doc_id,
                    "precision": precision,
                    "recall": recall
                })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    from corpus import CORPUS
    evaluator = Evaluation(CORPUS)
    results = evaluator.get_queries_with_scores(10)
    
    # Save to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2,ensure_ascii=False)
    
    important_docs = evaluator.get_queries_with_important_docs(10)
    with open("important_docs.json", "w", encoding="utf-8") as f:
        json.dump(important_docs, f, indent=2, ensure_ascii=False)

    for query, scores in results.items():
        print(f"Query: {query}\nScores: {scores}\n")

    results_evaluation = evaluator.evaluate_no_rank_metric(LanguageModel(Corpus(), 0.2), n_queries=10)
    print(results_evaluation)
    
    results_evaluations = evaluator.evaluate_rank_metric(LanguageModel(Corpus(), 0.2), n_queries=10, limit=10)
    print(results_evaluations)
    