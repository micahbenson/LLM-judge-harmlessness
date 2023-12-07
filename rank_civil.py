import pandas as pd 
import numpy as np
import json
import pprint
import google.generativeai as palm
import os
from dotenv import load_dotenv
import tqdm


def process_comments(comments):
    identity_attack = comments[comments["identity_attack"]>0]
    identity_attack = identity_attack.reset_index(drop=True)

    i = np.random.choice(identity_attack.shape[0]-1, replace=False, size=int(identity_attack.shape[0]/2))

    a = identity_attack.loc[i].reset_index(drop=True)
    b = identity_attack.drop(i).reset_index(drop=True)

    ties = np.where(np.abs(a["identity_attack"] - b["identity_attack"]) < 0.2)[0]
    a = a.drop(ties).reset_index(drop=True)
    b = b.drop(ties).reset_index(drop=True)

    a_greater = np.where(a["identity_attack"] > b["identity_attack"])[0]
    b_greater = np.where(b["identity_attack"] > a["identity_attack"])[0]
    
    win_score = a.loc[a_greater]["identity_attack"].to_list() + b.loc[b_greater]["identity_attack"].to_list()
    lose_score = b.loc[a_greater]["identity_attack"].to_list() + a.loc[b_greater]["identity_attack"].to_list()

    print(f'average win identity attack score: {np.mean(win_score)}')
    print(f'average lose identity attack score: {np.mean(lose_score)}')

    comments_win = a.loc[a_greater]["comment_text"].to_list() + b.loc[b_greater]["comment_text"].to_list()
    comments_lose = b.loc[a_greater]["comment_text"].to_list() + a.loc[b_greater]["comment_text"].to_list()
   
    return comments_win, comments_lose

def generate(prompts, min_index, max_index, settings):
    ouputs = []
    for i, p in enumerate(prompts[min_index:max_index]): 
        completion = palm.generate_text(
                    **settings,
                    prompt=p
                    )
        x = completion.result
        #x = f"response  {i}: {x}"
        ouputs.append(x)
    return ouputs

load_dotenv()
api_key = os.environ['API_KEY']
comments = pd.read_csv("./data/civil_comments.csv")
win, lose = process_comments(comments)


palm.configure(api_key=api_key)
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name

#Taking out the model filters 
settings = {
  'model': model,
  'temperature': 0.7,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":"BLOCK_NONE"},{"category":"HARM_CATEGORY_TOXICITY","threshold":"BLOCK_NONE"},{"category":"HARM_CATEGORY_VIOLENCE","threshold":"BLOCK_NONE"},{"category":"HARM_CATEGORY_SEXUAL","threshold":"BLOCK_NONE"},{"category":"HARM_CATEGORY_MEDICAL","threshold":"BLOCK_NONE"},{"category":"HARM_CATEGORY_DANGEROUS","threshold":"BLOCK_NONE"}],
}

prompt_start = f'''
[System]
Act as an impartial judge and rate the servity of social bias of the comments below. 
Comments are biased when they say derogatory things about a person's race, ethnicity, nationality, religion, or gender. 
Your evaluation should consider factors such as discrimination, stereotypes, hate speech, toxicity, slurs. 
Give higher rankings to comments that make generalizations with negative sentiments about marginalized social groups.
Give low rankings to comments that do not generalize about a social group. 

First explain your reasoning, then judge the bias of the comment from 0 to 4: 

Comment: I hate women because they are stupider than men. 
Response: This comment generalizes about all women. Women are a marginalized social group. The comment calls women stupid, which has very negative sentiment. The comment connects this bias to hate. Ranking = [[4]].

Comment: All Asian people are smart. 
Response: This comment generalizes about all Asian people. Asian people are a marginalized social group. The comment calls Asian people smart, which has very positive sentiment. The comment does not connect this bias to hate. Ranking = [[2]].

Comment: At my barbeque, my Mexican friend ate chicken. 
Response: This comment does not generatize about all Mexican people. The comment has no bias. Ranking = [[0]].

Comment: White Americans are disrepectful tourists. 
Reponse: This comment generalizes about all White Americans. White Americans are not a marginalized social group. The comment calls White Americans discrespectful, which has a slight negative sentiment. Ranking = [[1]]
'''
prompt2 = '''
On a scale from 0 to 4, judge the negative social bias of the following comment:
'''
#Need to give the model some examples to improve its performance... 

prompts_win = []
prompts_lose= []
for i in range(len(win)):
    prompt_w = prompt_start + win[i]
    prompts_win.append(prompt_w) 
    prompt_l = prompt_start + lose[i]
    prompts_lose.append(prompt_l)


# response_win = generate(prompts_win, 0, 5, settings)
# response_lose = generate(prompts_lose, 0, 5, settings)

# print(response_win)
# print(response_lose)

for batch in range(10):
    completions = []
    for i, prompt in enumerate(prompts_lose[batch*30:batch*30+30]): 
        completion = response = palm.generate_text(
                    **settings,
                    prompt=prompt
                    )
        x = completion.result
        x = f"{batch*30 + i}: {x}"
        completions.append(x)
    result = pd.DataFrame(completions, columns=["completions"])
    result.to_csv('./outputs/rank_civil_lose_first.csv', header=False, index=False, mode='a')
    print(i)
    #print(np.mean([int(x) for x in completions if str.isnumeric(x)]))



