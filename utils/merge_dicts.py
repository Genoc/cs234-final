# from StackOverFlow
# Question 38987
def merge_dicts(*dicts):
  result = {}
  for dictionary in dicts:
    result.update(dictionary)
    
  return result