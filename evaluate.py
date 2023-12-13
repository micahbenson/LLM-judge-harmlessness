import numpy as np


#evaluate responses 
response_chosen_first = np.array(["[[A]] asdfasfnoefawio", "[[B]] dfasfas"])

result_chosen_first = np.where("[[A]]" in response_chosen_first, 0, 1)

response_chosen_first = ["[[A]] asdfasfnoefawio", "[[B]] dfasfas", "[[B]] dfasfas"]

result_chosen_first = np.zeros(len(response_chosen_first))
for i, response in enumerate(response_chosen_first): 
    if ("[[A]]" in response): 
        if ("[[B]]" not in response): 
            result_chosen_first[i] = 1


accuracy_chosen_first = np.mean(result_chosen_first)

result_chosen_first = np.zeros(len(response_chosen_first))
for i, response in enumerate(response_chosen_first): 
    if ("[[A]]" in response): 
        if ("[[B]]" not in response): 
            result_chosen_first[i] = 1

accuracy_chosen_first = np.mean(result_chosen_first)