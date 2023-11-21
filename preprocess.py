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
    prompt = f'''
    [System]
    Please act as an impartial judge and evaluate the quality of the responses provided by two
    AI assistants to the user prompt displayed below. You should choose the assistant that
    follows the user's instructions and answers the user's question better. Your evaluation
    should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
    and level of detail of their responses. 
    
    Begin your evaluation by comparing the two
    responses and provide a short explanation. Avoid any position biases and ensure that the
    order in which the responses were presented does not influence your decision. Do not allow
    the length of the responses to influence your evaluation. Do not favor certain names of
    the assistants. Be as objective as possible. After providing your explanation, output your
    final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
    if assistant B is better, and "[[C]]" for a tie.
    <|The Start of Assistant A's Conversation with User|>
    ### User:
    {question 1}
    ### Assistant A:
    {answer 1}
    ### User:
    {question 2}
    ### Assistant A:
    {answer 2}
    <|The End of Assistant A's Conversation with User|>
    <|The Start of Assistant B's Conversation with User|>
    ### User:
    {question 1}
    ### Assistant B:
    {answer 1}
    ### User:
    {question 2}
    ### Assistant B:
    {answer 2}
    <|The End of Assistant B's Conversation with User|>
        '''
    