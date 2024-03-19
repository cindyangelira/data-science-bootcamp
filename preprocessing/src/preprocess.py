import pandas as pd
import re
import pickle

class Processor():
  def __init__(self, 
              nominal_col:list, 
              ordinal_col:list, 
              num_col:list, 
              order_dict:dict,
              default_order:list,
              encoder_path:str):
    self.nominal_col = nominal_col
    self.ordinal_col = ordinal_col
    self.num_col = num_col
    self.order_dict = order_dict
    self.default_order = default_order
    self.encoder_path = encoder_path
    self.encoder = self.load_encoder()

  def transform(self, df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns= lambda x:self.col_preprocessing(x))
    df = self.num_processing(df)
    df = self.encode_ordinal(df)
    encoder_array = self.encoder.transform(df[self.nominal_col]).toarray()
    encoder_col = self.encoder.get_feature_names_out(self.nominal_col)
    df = pd.concat([df.drop(columns=self.nominal_col), pd.DataFrame(encoder_array, columns = encoder_col).astype(int)], axis = 1)
    return df

  def col_preprocessing(self, x) -> str:
    x = x.lower().replace('(e27', '') 
    x = re.sub(r'\(.*?\)', '', re.sub('%', 'percent', x)) 
    x = re.sub(r'\s+', '_', x.strip()) 
    x = re.sub(r'[^a-z_]', '', x)
    return x
  
  def num_processing(self, df:pd.DataFrame) -> pd.DataFrame:
    for col in self.num_col:
      df[col] = df[col].str.replace(':','.')
      df[col] = df[col].astype(float)
    return df

  def load_encoder(self):
    with open(self.encoder_path, 'rb') as f:
      return pickle.load(f)

  def encode_ordinal(self, df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy() 
    for col in df[self.ordinal_col]:
      if col not in self.order_dict:
        self.order_dict[col] = self.default_order
    for col, order in self.order_dict.items():
      df[col] = df[col].map({category: i for i, category in enumerate(order)})
    return df