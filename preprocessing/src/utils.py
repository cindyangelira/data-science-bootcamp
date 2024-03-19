import pandas as pd

def convert_col(df:pd.DataFrame):
  type_dict = {
      'umur' : int,
      'is_menikah' : int,
      'is_keturunan' : int,
      'is_merokok' : int
  }
  df = df.astype(type_dict, errors = 'ignore')
  return df