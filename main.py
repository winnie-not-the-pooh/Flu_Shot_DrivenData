
import numpy as np
import pandas as pd

import os
import logging

from datetime import datetime

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import OneHotEncoder

#python3 -c "import pandas as pd; df=pd.read_csv('training_set_features.csv'); print(df['education'].unique())"
# # +++++++++++ READ DATA +++++++++++++++++++++++++
base_folder = ''

df = pd.read_csv(os.path.join(base_folder, 'training_set_features.csv'))
df_label = pd.read_csv(os.path.join(base_folder, 'training_set_labels.csv'))
df_test = pd.read_csv(os.path.join(base_folder, 'test_set_features.csv'))
df = pd.merge(df_label, df, on="respondent_id", how="inner")

LABELS = ['h1n1_vaccine', 'seasonal_vaccine'] # multilabel problem
CURRENT_Y = ['seasonal_vaccine'] 
ID = ['respondent_id']

# ++++++++++ DEFINE FEATURES +++++++++++++++++++++++++
def define_features(df):
    # encode features
    # standard scale
    # imbalanced dataset?
    # why would you combine features?

    # binary_features = ['sex', 'marital_status', 'rent_or_own']

    # for col in binary_features:
    #     df[col] = df[col].map({'no': 0, 'yes': 1})

    ordinal_features = [
        'age_group', 'education' 
        ,'income_poverty','rent_or_own'
        , 'census_msa']
    
    ohe_features = ['race', 'sex', 'employment_status', 'hhs_geo_region', 'employment_industry', 'employment_occupation']
    
    numeric_features = [col for col in df.columns 
                    if col not in ordinal_features + ohe_features + ID + LABELS]
    
    df[ordinal_features] = df[ordinal_features].fillna(-1)
    df[numeric_features] = df[numeric_features].fillna(-1)

#     # standard scaling for numerical features
#     # scaler = StandardScaler()
#     # scaled_array = scaler.fit_transform(df[numeric_features])
#     # scaled_df = pd.DataFrame(scaled_array, columns=numeric_features, index=df.index)
#     # df = pd.concat([df.drop(columns=numeric_features), scaled_df], axis=1)
  
    # manual mapping -------------

    feature_orders = {

        "age_group" : {
            '18 - 34 Years': 1,
            '35 - 44 Years': 2,
            '45 - 54 Years': 3,
            '55 - 64 Years': 4,
            '65+ Years': 5
            },

        "education" : {
            '< 12 Years': 0,
            '12 Years': 1,
            'Some College': 2,
            'College Graduate': 3
        },

        "income_poverty" : {
            'Below Poverty': 1,
            '<= $75,000, Above Poverty': 2,
            '> $75,000': 3
        },

        "marital_status" : {
            'Not Married' : 0,
            'Married' : 1
        },

        "rent_or_own" : {
            'Rent' : 0,
            'Own' : 1
        },

        "census_msa" : { 
            'Non-MSA' : 0,
            'MSA, Not Principle  City' : 1,
            'MSA, Principle City' : 2
        }   
    }
 
    for feature, mapping in feature_orders.items():
        df[feature] = df[feature].map(mapping).fillna(-1)

    # -----------------------------------
    # OHE 

    df_ohe = df[ohe_features]
    
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df_ohe)

    encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(df_ohe.columns),
    index=df_ohe.index
    ).astype("int8") #saves space 

    df = df.drop(columns=ohe_features)
    df = pd.concat([df, encoded_df], axis=1)

    return df

df = define_features(df)

y = df[CURRENT_Y]
X = df.drop(columns=ID+LABELS, inplace=True)

df_test = define_features(df_test)
out_ids = df_test[ID]
X_t = df_test.drop(columns=ID)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# use larger test size if dataset is smaller

# # Paramter tuning -----------------------------

# # param_dist = {
# #     'num_leaves': [75, 1100, 250],
# #     'learning_rate': [0.01, 0.05, 0.1],
# #     'n_estimators': [500, 750, 1000, 2000],
# #     'max_depth': [15, 20, 25],
# #     'min_child_samples': [10, 15, 20]
# # }

# # print("Starting hyperparameter tuning at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# # mdl = LGBMClassifier(objective='binary', random_state=42, verbose=-1,  scale_pos_weight = (659512.0 / 90488.0) )

# # # search = RandomizedSearchCV(
# # #     XGBClassifier(
# # #     objective="binary:logistic",
# # #     eval_metric="logloss",
# # #     use_label_encoder=False,
# # #     scale_pos_weight = (659512.0 / 90488.0),
# # #     verbose = -1
# # #     ), 
# # #     param_distributions=param_dist,
# # #     n_iter=20, cv=5, scoring='roc_auc', verbose=-1, n_jobs=-1, random_state=42
# # # )

# # search = RandomizedSearchCV(
# #     mdl, param_distributions=param_dist,
# #     n_iter=20, cv=5, scoring='roc_auc', verbose=0, n_jobs=-1, random_state=42
# # )
# # search.fit(X_train, y_train)

# # print("Best params:", search.best_params_)
# # print("Best CV ROC AUC:", search.best_score_)

# # # # Models -------------------------

# # Are the classes imbalanced? # YES - unbalanced 1/0 -> 0.27 h1n1; 0.87 seasonal
counts = y.value_counts()

model = XGBClassifier(
    random_state=42,
    objective = "binary:logistic", 
    eval_metric = 'auc',
    learning_rate = 0.1, 
    max_depth = 10, 
    subsample = 0.9, 
    colsample_bytree = 0.7,
    n_estimators= 2805,
    scale_pos_weight = (counts[0] / counts[1]) #negative/positive
) 

# model_lgb = LGBMClassifier(
#                         random_state=42, 
#                         eval_metric = 'auc',
#                         objective = "binary",
#                         boosting_type= 'gbdt',
#                         num_leaves=31, 
#                         n_estimators=1000, 
#                         min_child_samples=20,
#                         max_depth=15,
#                         learning_rate=0.05,
#                         verbose=-1,
#                         scale_pos_weight = (659512.0 / 90488.0) 
# ) 
# # # AUC 0.81752 no params, 0.83407 with num_leaves=100, n_estimators=500, min_child_samples=20,max_depth=20,learning_rate=0.05

# # ## predict 
model.fit(X_train,y_train)
model.fit(X_train, y_train)

# # calculate AUC -------------------
y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC:", roc_auc)
# #-----------------------------------

# # ######################################
# slider = 0.8

# y_t = model.predict(X_t)
# y_lgb = model_lgb.predict(X_t)
# y_t = slider*y_t + (1-slider)*y_lgb 


# out_df = pd.DataFrame({
#     'id': out_ids,
#     'predicted_target': y_t
# })

# out_df.to_csv('oct5.csv', index=False)   