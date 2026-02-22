import pandas as pd
from markov import MarkovRecommender

data = {
    "session_id": [1,1,1,2,2,2],
    "item_id": ["A","B","C","A","B","D"],
    "timestamp": [1,2,3,1,2,3]
}

df = pd.DataFrame(data)

model = MarkovRecommender()
model.fit(df)

print(model.transition_probs)
print(model.recommend("B"))