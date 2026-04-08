import os
import sys
import pandas as pd
from keras.models import load_model

sys.path.append("code")
import tools as t


MODEL_PATH = "./models/spam_detector_nlp.keras" # Path to the model
TOKENIZER_PATH = "./models/tokenizer.pk1"       # Path to the tokenizer


def read_files():
    """
      Reads files for prediction.

      Returns:
          list: list of pandas DataFrames
          list: names of files
    """
    data = []
    names = []
    dir_entries = os.listdir(".")                                                      # List of files in current directory
    data_files = [f for f in dir_entries if f.endswith(".csv")]                        # List of .csv files in current directory
    for f in data_files:
        try:
            data.append(pd.read_csv(f, header=None).iloc[:, [0]])                      # Reading .csv file and adding it to data_files
        except Exception as e:
            print(f"An error occured while reading file {f}: {type(e).__name__}: {e}")
        else: 
            print(f"Found {f}")

            names.append(f)                                                            # Adding file name to names
    return data, names
        

def predict(data):
    """
      Predicts labels.

      Args:
          data (pandas DataFrame): data for prediction

      Returns:
          pandas DataFrame: prediction result
    """
    nlp_model = load_model(MODEL_PATH)                                                        # Loading model
    X, _ = t.tokenizing(data.copy(), text_col=data.columns[0], tokenizer_path=TOKENIZER_PATH) # Tokenizing data
    prob = nlp_model.predict(X, batch_size=1).reshape(-1)                                     # Predicting probabilities
    prediction = (nlp_model.predict(X, batch_size=1) > 0.5).astype(int).reshape(-1)           # Predicting labels
    result = pd.DataFrame({"data": data.iloc[:, 0],
                           "prediction": prediction,
                           "probability": prob})
    return result
    

def write_prediction(prediction, name):
    """
      Writes prediction into .csv

      Args:
          prediction (pandas DataFrame): prediction
          name (string): name of file for prediction
    """
    prediction.to_csv(f"prediction_{name}", index=False)
    
    
if __name__ == "__main__":
    data, names = read_files()
    if not data:
        print("No files found")
        exit(1)
    
    for inx, df in enumerate(data):
        prediction = predict(df.copy())
        write_prediction(prediction, names[inx])
    exit(0)
