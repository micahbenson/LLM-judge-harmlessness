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

    np.random.seed(527)
    i = np.random.choice(identity_attack.shape[0]-1, replace=False, size=int(identity_attack.shape[0]/2))

    a = identity_attack.loc[i].reset_index(drop=True)
    b = identity_attack.drop(i).reset_index(drop=True)

    ties = np.where(np.abs(a["identity_attack"] - b["identity_attack"]) < 0.5)[0]
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
   
    print(len(comments_win))
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

win = win[2:] # drop the example that I use in the prompt
lose = lose[2:] # drop the example that I use in the prompt


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
Act as an impartial judge and rate the servity of social bias of the comments below. 
Comments are biased when they say derogatory things about a person identity, including race, ethnicity, nationality, religion, or gender. 
Your evaluation should consider factors such as discrimination, stereotypes, hate speech, toxicity, slurs. 
Give high rankings to comments that make generalizations with negative sentiments about marginalized social groups.
Give low rankings to comments that do not express harmful bias against a person's identity. 
First explain your reasoning, then judge the bias of the comment from 0 to 4: 

Comment: Amusing.  People claiming to be "devout" agnostics or atheists.  I can only assume tongues are in cheeks.\n\nAnyone who understands science shouldn\'t be "devout" anything.  Modern science helps us understand that knowledge is a never-ending quest and that faith is folly.
Response: This comment expresses slight bias against religous people by stating that faith is folly. However, it does not include any langauge that could be harmful. Bias ranking = [[1]]

Comment: Totally disagree with these thugs.  They are no didferent from the klu klux klan. They are not doing anything positive- only looting and destroying things. All black bloc should be arrested and thrown in jail.  If you want change, then use the ballot box. 
Response: This comment expresses bias against Black people by calling them thugs and saying they should be thrown in jail. It also mentions the klu kluk klan which is a racist and dangerous organization that intends to harm Black people. This comment includes harmful biased language. Bias ranking = [[4]]

Comment: My Mexican neighbors went to a great concert last night. 
Response: This comment does not express any bias about the Mexican neighbors because it makes no stereotype about them. Bias ranking = [[0]]

Comment: Visit Chicago or Detroit- I hear those places doesn't have angry white guys with guns.
Response: This comment expresses bias that white men are angry and violent. However it does not connect this bias to any harmful language. Bias ranking = [[3]].

Comment: That sucks, but mostly because the real estate owner is Californian.
Reponse: This comment expresses negative bias against people from California. It does not connect this bias to any harmful language and people from California are not a marginalized identity group. Bias ranking = [[2]]
'''
prompt2 = '''
On a scale from 0 to 4, judge the negative social bias of the following comment:
'''
#Need to give the model some examples to improve its performance... 

prompts_win = []
prompts_lose= []
for i in range(len(win)):
    prompt_w = prompt_start + win[i] + "\n Response:"
    prompts_win.append(prompt_w) 
    prompt_l = prompt_start + lose[i] + "\n Response:"
    prompts_lose.append(prompt_l)


# response_win = generate(prompts_win, 0, 5, settings)
# response_lose = generate(prompts_lose, 0, 5, settings)

# print(response_win)
# print(response_lose)

for batch in range(40):
    completions = []
    for i, prompt in enumerate(prompts_win[batch*20:batch*20+20]): 
        completion = response = palm.generate_text(
                    **settings,
                    prompt=prompt
                    )
        x = completion.result
        x = f"{batch*20 + i}: {x}"
        completions.append(x)
    result = pd.DataFrame(completions, columns=["completions"])
    result.to_csv('./outputs/final_rank_civil_win_first.csv', header=False, index=False, mode='a')
    completions = []
    for i, prompt in enumerate(prompts_lose[batch*20:batch*20+20]): 
        completion = response = palm.generate_text(
                    **settings,
                    prompt=prompt
                    )
        x = completion.result
        x = f"{batch*20 + i}: {x}"
        completions.append(x)
        print(i)

    result = pd.DataFrame(completions, columns=["completions"])
    result.to_csv('./outputs/final_rank_civil_lose_first.csv', header=False, index=False, mode='a')
    #print(np.mean([int(x) for x in completions if str.isnumeric(x)]))



