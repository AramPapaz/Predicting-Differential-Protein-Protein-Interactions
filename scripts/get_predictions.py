import pandas as pd
import numpy as np
from collections import Counter,defaultdict
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import argparse



def predict_final(X,models):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


parser = argparse.ArgumentParser()
parser.add_argument("MSFeatures_Table", help="MS features computed for a PPI.")
parser.add_argument("Embeddings_Table", help="Mint embeddings for each PPI.")
parser.add_argument("RFModelPath", help="Path of the weighted RF model.")
parser.add_argument("Output_filename", help="Output file name.")

args = parser.parse_args()


ms_data=pd.read_csv(args.MSFeatures_Table)

embd_data=pd.read_csv(args.Embeddings_Table)


loaded_model = joblib.load(args.RFModelPath)



finaldata=ms_data.merge(embd_data,on='Pair',how='inner').reset_index(drop=True)
print(finaldata)

X=finaldata.drop(['Pair'],axis=1)

preds=predict_final(X,loaded_model)

res=pd.DataFrame({'Pair':list(finaldata['Pair']),'Prediction':list(preds)})
res=res.sort_values(by='Prediction',ascending=False).reset_index(drop=True)
res.to_csv(args.Output_filename,index=False)



