import os
import sys
import pandas as pd
from keras.models import load_model

sys.path.append("code")
import tools as t


MODEL_PATH = "./models/spam_detector_nlp.keras"
TOKENIZER_PATH = "./models/tokenizer.pk1"


def read_files():
    data = []
    names = []
    dir_entries = os.listdir(".")
    data_files = [f for f in dir_entries if f.endswith(".csv")]
    for f in data_files:
        try:
            data.append(pd.read_csv(f, header=None).iloc[:, [0]])
        except Exception as e:
            print(f"An error occured while reading file {f}: {type(e).__name__}: {e}")
        else: 
            print(f"Found {f}")

            names.append(f)
    return data, names
        

def predict(data):
    nlp_model = load_model(MODEL_PATH)
    X, _ = t.tokenizing(data.copy(), text_col=data.columns[0], tokenizer_path=TOKENIZER_PATH)
    prob = nlp_model.predict(X, batch_size=1).reshape(-1)
    prediction = (nlp_model.predict(X, batch_size=1) > 0.5).astype(int).reshape(-1)
    result = pd.DataFrame({"data": data.iloc[:, 0],
                           "prediction": prediction,
                           "probability": prob})
    return result
    

def write_prediction(prediction, name):
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
