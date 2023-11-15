import pandas as pd
import numpy as np

df = pd.read_json("./data/train.jsonl", lines=True)

for row in len(df.rows):
    chosen = df.iloc[row, 0]
    rejected = df.iloc[row, 1]

    first = np.random.randint(2)
    query1_true = 0
    if first==0:
        test1 = (chosen, rejected)
        query1_true = 1
    else: 
        test1 = (rejected, chosen)
        query1_true = 2
    
    #create the prompt send query to GPT 4... 
    prompt = f""