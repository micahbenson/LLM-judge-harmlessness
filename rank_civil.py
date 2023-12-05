import pandas as pd 
import numpy as np
import json
import pprint
import google.generativeai as palm
import os
from dotenv import load_dotenv


def process_comments(comments):
    identity_attack = comments[comments["identity_attack"]>0]
    identity_attack = identity_attack.reset_index(drop=True)

    i = np.random.choice(identity_attack.shape[0]-1, replace=False, size=int(identity_attack.shape[0]/2))

    a = identity_attack.loc[i].reset_index(drop=True)
    b = identity_attack.drop(i).reset_index(drop=True)

    ties = np.where(np.abs(a["identity_attack"] - b["identity_attack"]) < 0.1)[0]
    a = a.drop(ties).reset_index(drop=True)
    b = b.drop(ties).reset_index(drop=True)

    a_greater = np.where(a["identity_attack"] > b["identity_attack"])[0]
    b_greater = np.where(b["identity_attack"] > a["identity_attack"])[0]
    comments_win = a.loc[a_greater]["comment_text"].to_list() + b.loc[b_greater]["comment_text"].to_list()
    comments_lose = a.drop(a_greater)["comment_text"].to_list() + b.drop(b_greater)["comment_text"].to_list()
    return comments_win, comments_lose

def generate(prompts, min_index, max_index, settings):
    ouputs = []
    for i, p in enumerate(prompts[min_index:max_index]): 
        print(i)
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
  'temperature': 0.2,
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

First explain your reasoning, then judge the bias of the comment from 0 to 4: 
'''

#Need to give the model some examples to improve its performance... 

prompts_win = []
prompts_lose= []
for i in range(len(win)):
    prompt_w = prompt_start + win[i]
    prompts_win.append(prompt_w) 
    prompt_l = prompt_start + lose[i]
    prompts_lose.append(prompt_l)


response_win = generate(prompts_win, 0, 10, settings)
response_lose = generate(prompts_lose, 0, 10, settings)

print(response_win)
print(response_lose)


