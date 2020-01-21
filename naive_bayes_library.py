import pandas as pd

def bayes(evidence:set, evidence_bag:dict, training_table:pd.DataFrame) -> tuple:
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
  probablility_tuple = tuple(probability_list)
  return probablility_tuple

def bayes_tester(testing_table:pd.DataFrame, evidence_bag:dict, training_table:pd.DataFrama, parser:callable) -> list:
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