import pandas as pd
import numpy as np
import spacy
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from typing import TypeVar


dframe = TypeVar('pd.core.frame.DataFrame')
swords = stopwords.words('english')
swords.sort()

def ordered_distances(target_vector:list, crowd_table, answer_column:str, dfunc) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  crowd_copy = crowd_table.drop(answer_column, axis=1) # Would it be bad to 
  distance_list = []                                   # drop in-place instead? (yes)
  for index, row in crowd_copy.iterrows():
    row_vector = [row[i] for i in range(row.size)]
    distance = dfunc(row_vector, target_vector)
    distance_list.append((index, distance))
  
  distance_list.sort(key = lambda dist_tup: dist_tup[1])
  return distance_list

def knn(target_vector:list, crowd_table, answer_column:str, k:int, dfunc) -> int:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  polled = ordered_distances(target_vector, crowd_table, answer_column, dfunc)[:k]
  polled_id = [i for i,d in polled]
  polled_survival = [crowd_table.loc[i, answer_column] for i in polled_id]
  return int(sum(polled_survival)/len(polled_survival) >= .5)
  
def knn_accuracy(test_table, crowd_table, answer_column:str, k:int, dfunc) -> float:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  possible_points = test_table.shape[0]
  points = 0
  for index, row in test_table.iterrows():
    answer_col_val = row[answer_column]
    row.drop(answer_column, inplace=True)
    target_vector = [row[i] for i in range(row.size)]
    prediction = knn(target_vector, crowd_table, answer_column, k, dfunc)
    match = prediction == answer_col_val
    points += int(match)

  return points/possible_points
  
def knn_tester(test_table, crowd_table, answer_column, k, dfunc) -> dict:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  condition_count = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  for index, row in test_table.iterrows():
    outcome = row[answer_column]
    row.drop(answer_column, inplace=True)
    target_vector = [row[i] for i in range(row.size)]
    prediction = knn(target_vector, crowd_table, answer_column, k, dfunc)
    condition_count[(prediction, outcome)] += 1

  return condition_count

def cm_accuracy(condition_count):
  total_correct = condition_count[(0,0)] + condition_count[(1,1)]
  total = condition_count[(0,0)] + condition_count[(1,1)] + condition_count[(1,0)] + condition_count[(0,1)]
  return total_correct/total

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  if not any(vect1) or not any(vect2):
    return 0
  product = 0
  magnitude_vect1 = 0
  magnitude_vect2 = 0
  for index, component in enumerate(vect1):
    component2 = vect2[index]
    product += component2*component
    magnitude_vect1 += component**2
    magnitude_vect2 += component2**2
  magnitude_vect1 = magnitude_vect1**(1/2)
  magnitude_vect2 = magnitude_vect2**(1/2)
  return product/(magnitude_vect1*magnitude_vect2)

def inverse_cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f'Mismatching length for vectors: {len(vect1)} and {len(vect2)}'
  normal_result = cosine_similarity(vect1, vect2)
  return 1.0 - normal_result

def get_clean_words(stopwords:list, raw_sentence:str) -> list:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(raw_sentence, str), f'raw_sentence must be a str but saw a {type(raw_sentence)}'

  sentence = raw_sentence.lower()
  for word in stopwords:
    sentence = re.sub(r"\b"+word+r"\b", '', sentence)  #replace stopword with empty

  cleaned = re.findall("\w+", sentence)  #now find the words
  return cleaned

def bayes(evidence:set, evidence_bag:dict, training_table):
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  probability_list = [1 for label in list(evidence_bag.values())[0]]
  label_list = training_table['label'].values.tolist()
  label_counts = (label_list.count(label) for label, probability in enumerate(probability_list)) # Pieces of data per label

  for label, count in enumerate(label_counts):
    for item in evidence:
      probability_list[label] *= evidence_bag[item][label]/count
    probability_list[label] *= count/len(training_table)
  #probablility_tuple = tuple(probability_list)
  return probablility_list

def bayes_tester(testing_table: dframe, evidence_bag:dict, training_table, parser:callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'
  bayes_values = []
  for _, row in testing_table.iterrows():
    parsed = set(parser(row['text'])) 
    bayes_values.append(bayes(parsed, evidence_bag, training_table))
  return bayes_values

def robust_bayes(evidence:set, evidence_bag:dict, training_table, laplace:float=1.0):
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert isinstance(laplace, float), f'laplace not a float but instead a {type(laplace)}'
  probability_list = [1 for label in list(evidence_bag.values())[0]]
  label_list = training_table['label'].values.tolist()
  label_counts = (label_list.count(label) for label, probability in enumerate(probability_list)) # Pieces of data per label

  for label, count in enumerate(label_counts):
    for item in evidence:
      probability_list[label] *= (evidence_bag.get(item, [0,0,0])[label] + laplace)/(len(evidence_bag) + count + 1)
    probability_list[label] *= count/(len(training_table))
  probablility_tuple = tuple(probability_list)
  return probability_tuple

def euclidean_distance(vect1, vect2):
  square_distance = 0
  for index, element in enumerate(vect1):
    square_distance += (element - vect2[index]) ** 2
  return square_distance ** 0.5

def fast_euclidean_distance(x, y) -> float:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  return sum(np.square(np.subtract(x, y))) ** .5

def subtractv(x:narray, y:narray) -> narray:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  return np.subtract(x, y) # Is this cheating?

def addv(x:narray, y:narray) -> narray:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  return np.add(x, y)

def meanv(matrix: narray) -> narray:
  assert isinstance(matrix, numpy.ndarray), f"matrix must be a numpy array but instead is {type(matrix)}"
  assert len(matrix.shape) == 2, f"matrix must be a 2d array but instead is {len(matrix.shape)}d"
  return np.array(np.mean(matrix, axis=0).flat)

from numpy.linalg import norm  #hint: i found this useful

def fast_cosine(v1:narray, v2:narray) -> float:
  assert isinstance(v1, numpy.ndarray), f"v1 must be a numpy array but instead is {type(v1)}"
  assert len(v1.shape) == 1, f"v1 must be a 1d array but instead is {len(v1.shape)}d"
  assert isinstance(v2, numpy.ndarray), f"v2 must be a numpy array but instead is {type(v2)}"
  assert len(v2.shape) == 1, f"v2 must be a 1d array but instead is {len(v2.shape)}d"
  assert len(v1) == len(v2), f'v1 and v2 must have same length but instead have {len(v1)} and {len(v2)}'
  if not np.any(v1) or not np.any(v2):
    return 0.0
  return np.dot(v1, v2)/(norm(v1)*norm(v2))

def dict_ordered_distances(space:dict, coord:narray) -> list:
  assert isinstance(space, dict), f"space must be a dictionary but instead a {type(space)}"
  assert isinstance(list(space.values())[0], numpy.ndarray), f"space must have numpy arrays as values but instead has {type(space.values()[0])}"
  assert isinstance(coord, numpy.ndarray), f"coord must be a numpy array but instead is {type(cord)}"
  assert len(list(space.values())[0]) == len(coord), f"space values must be same length as coord"
  assert len(coord) == 3, "coord must be a triple"
  distances = []
  for key, value in space.items():
    distances.append((key, fast_euclidean_distance(value, coord)))
  return sorted(distances, key = lambda x: x[1])

def vec(nlp, s:str):
    return nlp.vocab[s].vector

def sent2vec(nlp:spnlp, s: str):
  word_array = []
  for token in nlp(s.lower()):
    token = token.text
    word_array.append(vec(nlp, token))
  return meanv(np.matrix(word_array))

def spacy_closest_sent(nlp, space:list, input_str:str, n:int=10):
  assert isinstance(space, list)
  assert all([isinstance(sp, spacy.tokens.span.Span) for sp in space])

  input_vec = sent2vec(nlp, input_str)
  return sorted(space,
                key=lambda x: fast_cosine(np.mean([w.vector for w in x], axis=0), input_vec),
                reverse=True)[:n]
