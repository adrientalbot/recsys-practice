import pandas as pd
from collections import defaultdict


class MarkovRecommender:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}

    def fit(self, df):
        df = df.sort_values(["session_id", "timestamp"])

        for session_id, group in df.groupby("session_id"):
            items = group["item_id"].tolist()

            for t in range(len(items) - 1):
                i = items[t]
                j = items[t + 1]
                self.transition_counts[i][j] += 1

        self._compute_probabilities()

    def _compute_probabilities(self):
        for i, next_items in self.transition_counts.items():
            total = sum(next_items.values())
            self.transition_probs[i] = {
                j: count / total
                for j, count in next_items.items()
            }

    def recommend(self, current_item, top_k=3):
        if current_item not in self.transition_probs:
            return []

        probs = self.transition_probs[current_item]
        return sorted(probs.items(), key=lambda x: -x[1])[:top_k]