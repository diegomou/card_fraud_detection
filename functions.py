import json
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path_to_data, drop_userID_and_cardnum=True):
    raw_data = pd.read_csv(path_to_data)
    if drop_userID_and_cardnum:
        output = raw_data.drop(columns=['User', 'Card'])
        cols = ['year', 'month', 'day', 'time', 'amount', 'use_chip',
       'merchant_name', 'merchant_city', 'merchant_state', 'zip', 'MCC',
       'errors', 'is_fraud']
    else:
        output = raw_data
        cols = ['user', 'card', 'year', 'month', 'day', 'time', 'amount', 'use_chip',
       'merchant_name', 'merchant_city', 'merchant_state', 'zip', 'MCC',
       'errors', 'is_fraud']
    output.columns = cols

    return output

def missing_values_assesment(df):
    N_tot = len(df[df.columns[0]])
    for column in df.columns:
        print("#---------------------------------------#")
        N_missing = len(df[column][df[column].isna()])
        perc_missing = np.round(100*N_missing/N_tot, 3)
        print(f"Nulls values in {column}: {N_missing}")
        print(f"Percentage of missing values in {column}: {perc_missing}%")

def data_adequation(df, data_types_dict=None):
    if data_types_dict is not None:
        data_types_dict = json.load(open(data_types_dict))
        output = df.astype(data_types_dict)
 
    #Ad hoc modifications
    #parse_weekday = lambda row: pd.to_datetime("{year}-{month}-{day}".format(year=row.year, month=row.month, day=row.day)).weekday()
    parse_amout = lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", x)[0])
    parse_hour = lambda x : int(x.hour)
    
    output['amount'] = output['amount'].apply(parse_amout)
    output["hour"] = output["time"].apply(parse_hour)
    #output['weekday'] = output.apply(parse_weekday, axis=1)  
    output["errors"] = output["errors"].replace("nan", "unknown")
    return output
        
def hist_calculator(df, column, feature_type, label_col="is_fraud", label_values=["Yes", "No"]):
    hist_dict = dict()
    if feature_type == "int64" or feature_type == "object" or feature_type == "int32":
        bins = list(df[column].unique())
        bins.sort()
        for label in label_values:
            print(label)
            df_aux = df[df[label_col]==label]
            hist_values = list()
            N_label = len(df_aux)
            print(N_label)
            for value in bins:
                hist_values.append(df_aux[column][df_aux[column] == value].count()/N_label)
            hist_dict[label] = pd.DataFrame({'values':bins, 'prob':hist_values})
        print(hist_dict.keys())
    elif feature_type == "float64":
        bins = np.linspace(int(np.round(df[column].min())), 
                           int(np.round(df[column].max())) + 1,
                           400)
        for label in label_values:
            df_aux = df[df[label_col]==label]
            N_label = len(df_aux)
            print(N_label)
            values = list()
            hist_values = list()
            for i in range(len(bins)-1):
                value = int(0.5*(bins[i] + bins[i+1]))
                values.append(value)
                if i < len(bins)-1:
                    hist_values.append(df_aux[column][(df_aux[column] >= bins[i]) & (df_aux[column] < bins[i+1])].count()/N_label)
                else:
                    hist_values.append(df_aux[column][(df_aux[column] >= bins[i]) & (df_aux[column] <= bins[i+1])].count()/N_label)
            hist_dict[label] = pd.DataFrame({'values':values, 'prob':hist_values})
        else:
            print(feature_type)
    return hist_dict

def plot_generator(df, column, feature_type):
        hist_dict = hist_calculator(df, column, feature_type)
        bins_fraud = hist_dict["Yes"]
        bins_not_fraud = hist_dict["No"]
        width_value = 0.4
        alpha_value = 0.6
        
        x = np.arange(len(np.array(hist_dict["No"]["values"])))
        plt.figure(column, figsize=(12,10))
        plt.bar(x - width_value*0.5, 
                np.array(bins_not_fraud.prob), 
                color='chartreuse',
                width=width_value, 
                alpha=alpha_value,
                label="No")
        plt.bar(x + width_value*0.5, 
                np.array(bins_fraud.prob), 
                color='crimson',
                width=width_value, 
                label="Yes",
                alpha=alpha_value)
        plt.xticks(x, bins_fraud["values"], rotation=90)
        if column == "amount":
            plt.xlim(xmin=0, xmax=40)
        plt.xlabel(f"{column}")
        plt.ylabel("")
        plt.title(f"Marginal distributions of empirical probability - {column}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
def features_plotter(df, cols_to_ignore=None):
    dataTypeDict = dict(df.dtypes)
    if cols_to_ignore is not None:
        df = df.drop(columns=cols_to_ignore)
    for column in df.columns:
        if column == "errors":
            plot_generator(df, column, feature_type=dataTypeDict[column])
            aux_df = df[df[column] != "unknown"]
            plot_generator(aux_df, column, feature_type=dataTypeDict[column])
        elif column == "is_fraud":
            pass
        else:
            plot_generator(df, column, feature_type=dataTypeDict[column])
            
def features_preprocessing(df, data_types, cols_to_ignore=None):
    if isinstance(data_types, str):
        output = data_adequation(df, data_types_dict=data_types)
    else:
        raise ValueError("Invalid type for data_types. It must be a string with the path to the data_types_dict")
    
    ok_values = ['unknown', 'Technical Glitch', 'Insufficient Balance', 'Bad PIN', 'Bad Expiration', 'Bad Card Number', 'Bad CVV', 'Bad Zipcode']
    output["errors"] = output["errors"].apply(lambda x: x if x in ok_values else "error_other")
    output["zip"] = output["zip"].fillna("unknown")
    output["merchant_state"] = output["merchant_state"].fillna("unknown")
    
    if isinstance(cols_to_ignore, list):
        output = output.drop(columns=cols_to_ignore)
    elif cols_to_ignore is None:
        pass
    else:
        raise ValueError("cols_to_ignore must be a list")
        
    return output

def dataset_undersampler(df, N_ratio=10):
    df_class_0 = df[df["is_fraud"] == "No"]
    df_class_1 = df[df["is_fraud"] == "Yes"]
    N_class_1 = df_class_1["is_fraud"].count()
    df_class_0_under = df_class_0.sample(int(N_class_1*N_ratio))
    
    output = pd.concat([df_class_1, df_class_0_under], ignore_index=True)
    
    return output

def TNR_score(cm):
    return cm[0][0]/(cm[0][0]+cm[0][1])

def pred_over_new_data(instance, path_to_model, data_types_path="./data_types_inference.json"):
    #Applying the features preprocessing function to the raw data
    instances_adecuated = features_preprocessing(instance, data_types=data_types_path)
    instances_adecuated = instances_adecuated.drop(columns=["time"])

    #Loading the trained model
    model =  pickle.load(open(path_to_model, 'rb'))
    
    #Making the prediction
    pred_label = model.predict(instances_adecuated)
    
    return pred_label