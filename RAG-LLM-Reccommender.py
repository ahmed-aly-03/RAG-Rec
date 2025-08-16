import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict
import ast
from openai import OpenAI

def create_embed_test_subsets_ML1M(ratings_fp, movies_fp):
  user_ratings = pd.read_csv(ratings_fp, usecols = [0,1,2], sep='::', names=['user_id', 'movie_id', 'rating'], engine='python')
  movie_names = pd.read_csv(movies_fp, usecols = [0,1], sep='::', names= ['movie_id', 'title'],engine='python', encoding = 'latin1')
  user_ratings = user_ratings.merge(movie_names, on='movie_id')
  user_ratings = user_ratings.drop(columns=['movie_id'])
  user_ratings = user_ratings[user_ratings['rating'] > 3]
  user_ratings = user_ratings.groupby("user_id")[["user_id", "title"]].apply(
    lambda df: f"User#{df['user_id'].iloc[0]} has liked:\n" + "\n".join(df["title"])
  ).reset_index(name="user rating history")
  embed_set, test_set = train_test_split(user_ratings, test_size=0.2, random_state=42)
  return embed_set, test_set

def get_ground_truth(ratings_fp, movies_fp):
  user_ratings = pd.read_csv(ratings_fp, usecols = [0,1,2], sep='::', names=['user_id', 'movie_id', 'rating'],
                             engine='python')
  movie_names = pd.read_csv(movies_fp, usecols = [0,1], sep='::', names= ['movie_id', 'title'],
                            engine='python', encoding = 'latin1')
  user_ratings = user_ratings.merge(movie_names, on='movie_id')
  user_ratings = user_ratings.drop(columns=['movie_id'])
  user_ratings = user_ratings[user_ratings['rating'] > 3]
  user_ratings  = user_ratings.groupby('user_id')['title'].apply(list).reset_index(name='ground_truth')
  return {row['user_id']: row['ground_truth'] for _, row in user_ratings.iterrows()}

def get_all_subsets(rating_fp, movies_fp):
  embed_set, test_set = create_embed_test_subsets_ML1M(rating_fp, movies_fp)
  ground_truth = get_ground_truth(rating_fp, movies_fp)
  return embed_set, test_set, ground_truth

def normalize_embeddings(x):
  return x/np.linalg.norm(x, axis=1, keepdims=True)

def generate_embedding(df):
  embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device = 'cuda')
  embeddings = []
  for _,row in df.iterrows():
    user_rating_history = row['user rating history']
    user_embedding = embedding_model.encode(user_rating_history)
    embeddings.append(user_embedding)
  normalized_embeddings = normalize_embeddings(np.array(embeddings))
  return normalized_embeddings,df['user_id'].tolist()

def get_similar_users(df, norm_embeddings, user_ids, top_k):
  results = []
  test_users,_ = generate_embedding(df)
  for j, user_vector in enumerate(test_users):
    similarities = np.dot(norm_embeddings, user_vector).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_sim_user_ids = [user_ids[i] for i in top_indices]
    results.append({'user vector': df.iloc[j]['user rating history'], 'top_similar_user_IDs': top_sim_user_ids})
  return results

def retrieve_similar_user_movies(df, user_id):
  user_rating_lookup = dict(zip(df['user_id'], df['user rating history']))
  user_rating_history = user_rating_lookup.get(user_id)
  return user_rating_history

def clean_similar_user_movies(similar_user_history):
    _, movies_str = similar_user_history.split(" has liked:\n")
    movies = movies_str.strip().split("\n")
    return movies

def get_similar_movies(top_similar_user_vectors, embed_set):
  similar_movies = []
  for dic in top_similar_user_vectors:
    similar_user_movies = set()
    for user_id in dic['top_similar_user_IDs']:
      user_movies = retrieve_similar_user_movies(embed_set, user_id)
      user_movies = clean_similar_user_movies(user_movies)
      similar_user_movies.update(user_movies)
    similar_movies.append({'user vector': dic['user vector'], 'similar_movies': similar_user_movies})
  return similar_movies

def retriver (df, norm_embeddings, user_ids, embed_set, top_k = 2):
  embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device = 'cuda')
  top_similar_user_vectors = get_similar_users(df, norm_embeddings, user_ids, top_k)
  retriver_output = get_similar_movies(top_similar_user_vectors, embed_set)
  return retriver_output

def generate_prompt(retriver_output, num_users, k = None):
  LLM_input = []
  for index, dic in enumerate(retriver_output):
    if index >= num_users:
      break
    movie_lines = dic['user vector'].splitlines()[1:]
    if len(movie_lines) < 20:
      continue
    user_movies = dic['user vector'].splitlines()[0:20]
    user_movies = '\n'.join(user_movies)
    movies = '\n'.join(sorted(dic['similar_movies'])[:len(movie_lines)])
    if k is not None:
      n = k
    else:
      n = len(movie_lines)
    similar_user_movies = 'Similar Users Liked Movies List:\n' + movies
    LLM_input.append([{"role": "system",
                       "content": f"You are a recommender system, from both lists generate an array of {n} recommendations based on genre/taste, don't include an explanation and return your results as a JSON array ('''json [ ]) You can include movies the user has watched already"},
                      {"role": "user",
                       "content": f"{user_movies}\n{similar_user_movies}"}])
  return LLM_input

def generate_recs_GPT4o(input_prompts, api_key):
  client = OpenAI(api_key=api_key)
  gpt_recs = []
  num_prompts = len(input_prompts)
  for i,prompt in enumerate(input_prompts):
    print(f'{i+1}/{num_prompts}')
    response = client.responses.create(model="gpt-4o",input = prompt)
    gpt_recs.append(response.output_text)
  return gpt_recs

def generate_recs_4o_mini(input_prompts, api_key):
  client = OpenAI(api_key=api_key)
  o4_mini_recs = []
  num_prompts = len(input_prompts)
  for i,prompt in enumerate(input_prompts):
    print(f'{i+1}/{num_prompts}')
    response = client.responses.create(model="gpt-4o-mini",input= prompt)
    o4_mini_recs.append(response.output_text)
  return o4_mini_recs

def clean_LLM_output(LLM_recs):
  cleaned_recs = []
  for rec in LLM_recs:
    code_block = re.search(r"```json\s*(.*?)```", rec, re.DOTALL)
    json_text = code_block.group(1).strip()
    cleaned_recs.append(json_text)
  return cleaned_recs

def calc_metrcics_LLM(cleaned_LLM_recs,retriver_out):
  precicsion_arr = []
  recall_arr = []
  f1_score_arr = []
  for i,rec in enumerate(cleaned_LLM_recs):
    predicted_results = ast.literal_eval(rec)
    ground_truth = retriver_out[i]['user vector'].splitlines()
    tp = len(set(predicted_results) & set(ground_truth))
    fp = len(set(predicted_results) - set(ground_truth))
    fn = len(set(ground_truth) - set(predicted_results))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    precicsion_arr.append(precision)
    recall_arr.append(recall)
    f1_score_arr.append(f1_score)
  precision_mean = np.mean(precicsion_arr)
  recall_mean = np.mean(recall_arr)
  f1_score_mean = np.mean(f1_score_arr)
  return {'precision:': precision_mean, 'recall:': recall_mean, 'F1_Score': f1_score_mean}

def calc_hit_rate_LLM(cleaned_LLM_recs,retriver_out,k):
  sum = 0
  for i,rec in enumerate(cleaned_LLM_recs):
    predicted_results = ast.literal_eval(rec)[:k]
    ground_truth = retriver_out[i]['user vector'].splitlines()
    tp = len(set(predicted_results) & set(ground_truth))
    if tp > 0:
      sum += 1
  return sum/len(cleaned_LLM_recs)

def calc_hit_ratek_LLM(cleaned_LLM_recs, retriver_output,k_arr):
  hit_rate_k = defaultdict(float)
  for k in k_arr:
    hit_rate_k[k] = calc_hit_rate_LLM(cleaned_LLM_recs,retriver_output,k)
  return hit_rate_k

def calc_hit_rate_RAG(retriver_output,k):
  sum = 0
  for output in retriver_output:
    ground_truth = sorted(output['user vector'].splitlines()[1:])
    rec_movies = sorted(output['similar_movies'])[:k]
    tp = len(set(ground_truth) & set(rec_movies))
    if tp > 0:
      sum += 1
  return sum/len(retriver_output)

def calc_hit_ratek_RAG(retriver_output,k_arr):
  hit_rate_k = defaultdict(float)
  for k in k_arr:
    hit_rate_k[k] = calc_hit_rate_RAG(retriver_output,k)
  return hit_rate_k

def calc_metrics_RAG(retriver_output):
  precicsion_arr = []
  recall_arr = []
  f1_score_arr = []
  for output in retriver_output:
    movie_lines = output['user vector'].splitlines()[1:]
    num_movies = len(movie_lines)
    if num_movies < 20:
      continue
    filtered_movies = sorted(output['similar_movies'])
    tp = len(set(movie_lines) & set(filtered_movies))
    fp = len(set(filtered_movies) - set(movie_lines))
    fn = len(set(movie_lines) - set(filtered_movies))
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    precicsion_arr.append(precision)
    recall_arr.append(recall)
    f1_score_arr.append(f1_score)
  precision_mean = np.mean(precicsion_arr)
  recall_mean = np.mean(recall_arr)
  f1_score_mean = np.mean(f1_score_arr)
  return {'precision:': precision_mean, 'recall:': recall_mean, 'F1_Score': f1_score_mean}