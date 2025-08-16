import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import ast
from openai import OpenAI
from collections import defaultdict

def create_test_set(ratings_fp, movies_fp):
  #Input: File paths to the movie lens ratings.dat and movie .dat files
  #Output: the set of users to test the model on 
  user_ratings = pd.read_csv(ratings_fp, usecols = [0,1,2], sep='::', names=['user_id', 'movie_id', 'rating'], engine='python')
  movie_names = pd.read_csv(movies_fp, usecols = [0,1], sep='::', names= ['movie_id', 'title'],engine='python', encoding = 'latin1')
  user_ratings = user_ratings.merge(movie_names, on='movie_id')
  user_ratings = user_ratings.drop(columns=['movie_id'])
  user_ratings = user_ratings[user_ratings['rating'] > 3]
  user_ratings = user_ratings.groupby("user_id")[["user_id", "title"]].apply(
    lambda df: f"User#{df['user_id'].iloc[0]} has liked:\n" + "\n".join(df["title"])
  ).reset_index(name="user rating history")
  _, test_set = train_test_split(user_ratings, test_size=0.2, random_state=42)
  return test_set

def create_prompt(test_set, num_users):
  #Input: Test set output from create_test_set() and the number of user desired to test the model on (test_set.length))
  #Output: Array with prompts ready to input into the model
  LLM_input = []
  for index,row in enumerate(test_set.iterrows()):
    if index >= num_users:
      break
    user_rating_history = row[1]['user rating history']
    n = user_rating_history.count('\n')
    LLM_input.append([
        {"role": "system",
         "content": f"You are a recommender system, based on the user's liked movies generate {n} recommendations, don't include an explanation and return your results as a JSON array ('''json [ ]). Only recommend movies in the movielens1M dataset"},
        {"role": "user", "content": user_rating_history}
    ])
  return LLM_input

def generate_recs_GPT4o(input_prompts, api_key):
  #Input: Arr of prompts (output of create_prompts()), and gpt api key
  #Output: recs for GPT4o model
  client = OpenAI(api_key=api_key)
  gpt_recs = []
  num_prompts = len(input_prompts)
  for i,prompt in enumerate(input_prompts):
    print(f'{i+1}/{num_prompts}')
    response = client.responses.create(model="gpt-4o",input = prompt)
    gpt_recs.append(response.output_text)
  return gpt_recs

def generate_recs_mini(input_prompts, api_key):
  #Input: Arr of prompts (output of create_prompts()), and gpt api key
  #output: recs for GPT4o mini model
  client = OpenAI(api_key=api_key)
  o4_mini_recs = []
  num_prompts = len(input_prompts)
  for i,prompt in enumerate(input_prompts):
    print(f'{i+1}/{num_prompts}')
    response = client.responses.create(model="gpt-4o-mini",input= prompt)
    o4_mini_recs.append(response.output_text)
  return o4_mini_recs

def clean_LLM_output(LLM_recs):
  #Input: Generated recs from the LLM
  #Output: Cleaned recs
  cleaned_recs = []
  for rec in LLM_recs:
    code_block = re.search(r"```json\s*(.*?)```", rec, re.DOTALL)
    json_text = code_block.group(1).strip()
    cleaned_recs.append(json_text)
  return cleaned_recs

def calc_metrics(cleaned_LLM_recs,test_set):
  #Input: arr of cleaned LLM output (output of clean_LLM_output()), test set to retrieve ground truth () (output of create_test_set())
  #Output: dic with the precision, recall and f-score for inputted test_set
  precision_arr = []
  recall_arr = []
  f1_score_arr = []
  for i,rec in enumerate(cleaned_LLM_recs):
    predicted_results = ast.literal_eval(rec)
    ground_truth = test_set.iloc[i]['user rating history'].splitlines()[1:]
    tp = len(set(predicted_results) & set(ground_truth))
    fp = len(set(predicted_results) - set(ground_truth))
    fn = len(set(ground_truth) - set(predicted_results))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    precision_arr.append(precision)
    recall_arr.append(recall)
    f1_score_arr.append(f1_score)
  precision_mean = np.mean(precision_arr)
  recall_mean = np.mean(recall_arr)
  f1_score_mean = np.mean(f1_score_arr)
  return {'precision:': precision_mean, 'recall:': recall_mean, 'F1_Score': f1_score_mean}

def calc_hit_rate(cleaned_LLM_recs,test_set,k):
  #Input: arr of cleaned LLM output (output of clean_LLM_output()), test set to retrieve ground truth (output of create_test_set()), k: top k user to calculate hit rate for
  #Output:Hit rate at k
  sum = 0
  for i,rec in enumerate(cleaned_LLM_recs):
    predicted_results = ast.literal_eval(rec)[:k]
    ground_truth = test_set.iloc[i]['user rating history'].splitlines()[1:]
    tp = len(set(predicted_results) & set(ground_truth))
    if tp > 0:
      sum += 1
  return sum/len(cleaned_LLM_recs)

def calc_hit_ratek_LLM(cleaned_LLM_recs, test_set,k_arr):
  #wrapper for calc_hit_rate, Input: K_arr (array with value to calculate hitrate@k for)
  #Ouput: Dic with hit rate 
  hit_rate_k = defaultdict(float)
  for k in k_arr:
    hit_rate_k[k] = calc_hit_rate(cleaned_LLM_recs,test_set,k)
  return hit_rate_k
