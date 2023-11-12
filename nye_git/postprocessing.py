import pandas as pd
import numpy as np
import preprocessing as pp
import re
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
#import a mean signed error function
def calculate_hourly_mae_and_plot(predictions, actuals):
    """
    Calculate the MAE for each hour of the day across multiple days, 
    and plot a histogram of these MAE values.
    Assumes each day contains 24 consecutive hourly observations in order.

    :param predictions: List or array of predictions.
    :param actuals: List or array of actual values.
    """
    if len(predictions) % 24 != 0 or len(actuals) % 24 != 0:
        raise ValueError("The length of predictions and actuals should be a multiple of 24.")

    hourly_mae = []
    num_days = len(predictions) // 24

    for hour in range(24):
        hourly_preds = predictions[hour::24]
        hourly_acts = actuals[hour::24]
        hourly_mae.append(mean_absolute_error(hourly_acts, hourly_preds))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(range(24), hourly_mae, color='skyblue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Absolute Error')
    plt.title('Hourly MAE of Predictions')
    plt.xticks(range(24), [f"{hour:02d}:00" for hour in range(24)])
    plt.grid(axis='y', linestyle='--')
    plt.show()


def calculate_hourly_me_and_plot(predictions, actuals):
    """
    Calculate the ME (Mean Error) for each hour of the day across multiple days, 
    and plot a histogram of these ME values.
    Assumes each day contains 24 consecutive hourly observations in order.

    :param predictions: List or array of predictions.
    :param actuals: List or array of actual values.
    """
    if len(predictions) % 24 != 0 or len(actuals) % 24 != 0:
        raise ValueError("The length of predictions and actuals should be a multiple of 24.")

    hourly_me = []
    num_days = len(predictions) // 24

    for hour in range(24):
        hourly_preds = predictions[hour::24]
        hourly_acts = actuals[hour::24]
        hourly_me.append(np.mean([a - p for a, p in zip(hourly_acts, hourly_preds)]))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(range(24), hourly_me, color='skyblue')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Error')
    plt.title('Hourly ME of Predictions')
    plt.xticks(range(24), [f"{hour:02d}:00" for hour in range(24)])
    plt.grid(axis='y', linestyle='--')
    plt.show()


def readRawTest(letter):
    df = pd.read_parquet(f"/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/{letter}/X_test_estimated.parquet")
    df.set_index("date_forecast", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def readAndBasicPreprocess(letter):
    X = readRawTest(letter)
    X.drop(columns=['cloud_base_agl:m'], inplace=True)
    X.drop(columns=['ceiling_height_agl:m'], inplace=True)
    X.drop(columns=['snow_density:kgm3'], inplace=True)
    X=pp.create_features(X)
    X=pp.create_time_features(X)
    X.drop(columns=['date_calc'], inplace=True)
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X = pp.add_lagged_features(X)
    return X

def makePrediction(A_model, B_model, C_model, filename):
    A_x_test = readAndBasicPreprocess("A")
    A_y_pred=A_model.predict(A_x_test)
    A_y_pred=pd.DataFrame(A_y_pred, index=range(0,720), columns=['prediction'])

    B_x_test=readAndBasicPreprocess("B")
    B_y_pred=B_model.predict(B_x_test)
    B_y_pred=pd.DataFrame(B_y_pred, index=range(720,1440), columns=['prediction'])

    C_x_test=readAndBasicPreprocess("C")
    C_y_pred=C_model.predict(C_x_test)
    C_y_pred=pd.DataFrame(C_y_pred, index=range(1440,2160), columns=['prediction'])

    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)

def makeEnsemblePrediction(A_xgb_model, A_xgb_processing, A_dnn_model, A_dnn_preprocessing, A_dnn_target_preprocessing, B_xgb_model, B_xgb_processing, B_dnn_model, B_dnn_preprocessing, B_dnn_target_preprocessing, C_xgb_model, C_xgb_processing, C_dnn_model,C_dnn_preprocessing, C_dnn_target_preprocessing, filename):
    A_X_test = readAndBasicPreprocess("A")

    A_X_test_xgb = A_xgb_processing.transform(A_X_test)
    A_y_pred_xgb = A_xgb_model.predict(A_X_test_xgb)

    A_X_test_dnn = pd.DataFrame(A_dnn_preprocessing.transform(A_X_test))
    A_y_pred_dnn = A_dnn_model.predict(A_X_test_dnn)
    A_y_pred_dnn = A_dnn_target_preprocessing.inverse_transform(A_y_pred_dnn).reshape(-1)

    A_y_pred = (A_y_pred_xgb + A_y_pred_dnn) / 2
    A_y_pred = pd.DataFrame(A_y_pred, index=range(0,720), columns=['prediction'])

    B_X_test = readAndBasicPreprocess("B")

    B_X_test_xgb = B_xgb_processing.transform(B_X_test)
    B_y_pred_xgb = B_xgb_model.predict(B_X_test_xgb)

    B_X_test_dnn = pd.DataFrame(B_dnn_preprocessing.transform(B_X_test))
    B_y_pred_dnn = B_dnn_model.predict(B_X_test_dnn)
    B_y_pred_dnn = B_dnn_target_preprocessing.inverse_transform(B_y_pred_dnn).reshape(-1)

    B_y_pred = (B_y_pred_xgb + B_y_pred_dnn) / 2
    B_y_pred = pd.DataFrame(B_y_pred, index=range(720,1440), columns=['prediction'])

    C_X_test = readAndBasicPreprocess("C")

    C_X_test_xgb = B_xgb_processing.transform(C_X_test)
    C_y_pred_xgb = B_xgb_model.predict(C_X_test_xgb)

    C_X_test_dnn = pd.DataFrame(B_dnn_preprocessing.transform(C_X_test))
    C_y_pred_dnn = B_dnn_model.predict(C_X_test_dnn)
    C_y_pred_dnn = B_dnn_target_preprocessing.inverse_transform(C_y_pred_dnn).reshape(-1)

    C_y_pred = (C_y_pred_xgb + C_y_pred_dnn) / 2

    C_y_pred = pd.DataFrame(C_y_pred, index=range(1440,2160), columns=['prediction'])

    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)
    

def make_dnn_prediction(A_model, A_preprocessing, A_target_scaling, B_model, B_preprocessing, B_target_scaling, C_model, C_preprocessing, C_target_scaling, filename):
    A_X_test = readAndBasicPreprocess("A")
    A_X_test_dnn = pd.DataFrame(A_preprocessing.transform(A_X_test))
    A_y_pred_dnn = A_model.predict(A_X_test_dnn)
    A_y_pred_dnn = A_target_scaling.inverse_transform(A_y_pred_dnn).reshape(-1)
    A_y_pred = pd.DataFrame(A_y_pred_dnn, index=range(0,720), columns=['prediction'])

    B_X_test = readAndBasicPreprocess("B")
    B_X_test_dnn = pd.DataFrame(B_preprocessing.transform(B_X_test))
    B_y_pred_dnn = B_model.predict(B_X_test_dnn)
    B_y_pred_dnn = B_target_scaling.inverse_transform(B_y_pred_dnn).reshape(-1)
    B_y_pred = pd.DataFrame(B_y_pred_dnn, index=range(720,1440), columns=['prediction'])

    C_X_test = readAndBasicPreprocess("C")
    C_X_test_dnn = pd.DataFrame(C_preprocessing.transform(C_X_test))
    C_y_pred_dnn = C_model.predict(C_X_test_dnn)
    C_y_pred_dnn = C_target_scaling.inverse_transform(C_y_pred_dnn).reshape(-1)
    C_y_pred = pd.DataFrame(C_y_pred_dnn, index=range(1440,2160), columns=['prediction'])

    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)


def make_xgb_prediction(A_model, A_preprocessing, B_model, B_preprocessing, C_model, C_preprocessing, filename):
    A_X_test = readAndBasicPreprocess("A")
    A_X_test_xgb = A_preprocessing.transform(A_X_test)

    A_y_pred_xgb = A_model.predict(A_X_test_xgb)
    A_y_pred = pd.DataFrame(A_y_pred_xgb, index=range(0,720), columns=['prediction'])

    B_X_test = readAndBasicPreprocess("B")
    B_X_test_xgb = B_preprocessing.transform(B_X_test)

    B_y_pred_xgb = B_model.predict(B_X_test_xgb)
    B_y_pred = pd.DataFrame(B_y_pred_xgb, index=range(720,1440), columns=['prediction'])

    C_X_test = readAndBasicPreprocess("C")
    C_X_test_xgb = C_preprocessing.transform(C_X_test)

    C_y_pred_xgb = C_model.predict(C_X_test_xgb)
    C_y_pred = pd.DataFrame(C_y_pred_xgb, index=range(1440,2160), columns=['prediction'])

    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)

def makeAutoMLPred(A_model, B_model, C_model, filename):
    """
    Assumes QuarterAsColumn

    """

    A_X_test = readAndBasicPreprocess("A")
    A_X_test = pp.QuartersAsColumnsTransformer().transform(A_X_test)
    A_y_pred = A_model.predict(A_X_test)
    A_y_pred = pd.DataFrame(A_y_pred, index=range(0,720), columns=['prediction'])

    B_X_test = readAndBasicPreprocess("B")
    B_X_test = pp.QuartersAsColumnsTransformer().transform(B_X_test)
    B_y_pred = B_model.predict(B_X_test)
    B_y_pred = pd.DataFrame(B_y_pred, index=range(720,1440), columns=['prediction'])

    C_X_test = readAndBasicPreprocess("C")
    C_X_test = pp.QuartersAsColumnsTransformer().transform(C_X_test)
    C_y_pred = C_model.predict(C_X_test)
    C_y_pred = pd.DataFrame(C_y_pred, index=range(1440,2160), columns=['prediction'])



    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.loc[(combined_pred.index % 24).isin([22, 23, 0]), "prediction"] = 0
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)

def compute_mae(ser1, ser2):
    """Compute Mean Absolute Error between two Series."""
    return np.abs(ser1 - ser2).mean()

def plot_mae_grid(dataframes_dict):
    """Plot a grid of MAE values for a dictionary of DataFrames."""
    
    labels = list(dataframes_dict.keys())
    dataframes = list(dataframes_dict.values())
    n = len(dataframes)
    
    mae_grid = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                mae_grid[i][j] = compute_mae(dataframes[i]["prediction"], dataframes[j]["prediction"])

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(mae_grid, cmap="viridis")
    
    ax.grid(False)
    plt.xticks(range(n), labels, rotation=45)
    plt.yticks(range(n), labels)
    
    # Add annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{mae_grid[i, j]:.2f}",
                           ha="center", va="center", color="w" if mae_grid[i, j] > (mae_grid.max() / 2) else "black")
    
    plt.colorbar(cax)
    plt.title('MAE Between DataFrames on "prediction" Column', pad=20)
    plt.show()


def test():
    print("hei")

def makePredictionWithModelAndPreprocessor(A_model, B_model, C_model, preprocessor, filename):
    """
    Assumes same preprocessing for all locations

    """

    A_X_test = readAndBasicPreprocess("A")
    A_X_test = preprocessor.transform(A_X_test)
    A_y_pred = A_model.predict(A_X_test)
    A_y_pred = pd.DataFrame(A_y_pred, index=range(0,720), columns=['prediction'])

    B_X_test = readAndBasicPreprocess("B")
    B_X_test = preprocessor.transform(B_X_test)
    B_y_pred = B_model.predict(B_X_test)
    B_y_pred = pd.DataFrame(B_y_pred, index=range(720,1440), columns=['prediction'])

    C_X_test = readAndBasicPreprocess("C")
    C_X_test = preprocessor.transform(C_X_test)
    C_y_pred = C_model.predict(C_X_test)
    C_y_pred = pd.DataFrame(C_y_pred, index=range(1440,2160), columns=['prediction'])

    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)


def submission_vs_best_submission(filepath):
    """
        mostly for debugging and testing. Checks if the submission is in the same ballpark as the best submission
    """

    refrence = pd.read_csv("/Users/henrikhorpedal/Documents/Skolearbeid/Maskinlæring/Group Task/TDT4173_Machine_Learning/Jallastacking/csvfiles/two_best_combined_zeroed_night_hours.csv")

    submission = pd.read_csv(filepath)

    print(f"MAE for location A: {mean_absolute_error(refrence['prediction'].iloc[0:720], submission['prediction'].iloc[0:720])}")
    print(f"MAE for location B: {mean_absolute_error(refrence['prediction'].iloc[720:1440], submission['prediction'].iloc[720:1440])}")
    print(f"MAE for location C: {mean_absolute_error(refrence['prediction'].iloc[1440:2160], submission['prediction'].iloc[1440:2160])}")


def make_average_prediction(preds_dict,filename):
    """
    Generates a prediction by taking the average of the predictions in preds_dict.
    """
    lenght = len(preds_dict)
    data = 0
    for value in preds_dict.values():
        data += value["prediction"]
    data = data / lenght
    data = pd.DataFrame(data, columns=['prediction'])
    data.index.name = "id"
    data["prediction"] = data['prediction'].apply(lambda x: 0 if x < 0.05 else x)
    data.loc[(data.index % 24).isin([22, 23, 0]), "prediction"] = 0
    data.to_csv(filename, index=True)



def make_lgbm_preprocessor_pred(A_model, A_scaler, A_preprocessor, B_model, B_scaler, B_preprocessor,C_model, C_scaler, C_preprocessor, filename):

    A_X_test = readAndBasicPreprocess("A")
    A_X_test = A_preprocessor.transform(A_X_test)
    A_X_test = A_scaler.transform(A_X_test)
    A_y_pred = A_model.predict(A_X_test)
    A_y_pred = pd.DataFrame(A_y_pred, index=range(0,720), columns=['prediction'])

    B_X_test = readAndBasicPreprocess("B")
    B_X_test = B_preprocessor.transform(B_X_test)
    B_X_test = B_scaler.transform(B_X_test)
    B_y_pred = B_model.predict(B_X_test)
    B_y_pred = pd.DataFrame(B_y_pred, index=range(720,1440), columns=['prediction'])

    C_X_test = readAndBasicPreprocess("C")
    C_X_test = C_preprocessor.transform(C_X_test)
    C_X_test = C_scaler.transform(C_X_test)
    C_y_pred = C_model.predict(C_X_test)
    C_y_pred = pd.DataFrame(C_y_pred, index=range(1440,2160), columns=['prediction'])

    combined_pred = pd.concat([A_y_pred, B_y_pred, C_y_pred], axis=0)
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    combined_pred.index.name = "id"
    combined_pred.to_csv(filename, index=True)

def make_one_location_pred(model, letter, preprocessor, filename):
    if letter == "A":
        index_range = range(0,720)
    elif letter == "B":
        index_range = range(720,1440)
    elif letter == "C":
        index_range = range(1440,2160)
    

    X_test = readAndBasicPreprocess(letter)
    X_test = preprocessor.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, index=index_range, columns=['prediction'])
    y_pred["prediction"] = y_pred["prediction"].clip(lower=0)
    y_pred.index.name = "id"
    y_pred.to_csv(filename, index=True)

def mean_diffrent_summers(filepath,folds_dict, predictionfilename):
    #FOLDS = {"A": 4, "B": 3, "C": 2}
    d_A = 0
    d_B = 0
    d_C = 0

    for letter in folds_dict.keys():
        preds = []
        folder_path = f"{filepath}/{letter}"
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):  # Check if the file is a CSV
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                data = df["prediction"]
                preds.append(data)
        #mean the predictions
        if letter == "A":
            for i,pred in enumerate(preds):
                if i == 0:
                    d_A = pred
                else:
                    d_A += pred
            d_A =d_A/len(preds)
            #d_A = pd.DataFrame(d_A, index=range(0,720), columns=['prediction'])

        
        elif letter == "B":
            for i,pred in enumerate(preds):
                if i == 0:
                    d_B = pred
                else:
                    d_B += pred
            #print(d_B)

            d_B = d_B/len(preds)
            #d_B = pd.DataFrame(d_B, index=range(720,1440), columns=['prediction'])

        
        elif letter == "C":
            for i,pred in enumerate(preds):
                if i == 0:
                    d_C = pred
                else:
                    d_C += pred
            d_C =d_C/len(preds)
            #d_C = pd.DataFrame(d_C, index=range(1440,2160), columns=['prediction'])

        
    combined_pred = pd.concat([d_A, d_B, d_C], axis=0)
    #name the column prediction
    combined_pred = pd.DataFrame(combined_pred, columns=['prediction'])
    combined_pred["prediction"] = combined_pred["prediction"].clip(lower=0)
    #reset index:
    combined_pred.reset_index(inplace=True, drop=True)
    combined_pred.index.name = "id"
    combined_pred.loc[(combined_pred.index % 24).isin([22, 23, 0]), "prediction"] = 0
    combined_pred.to_csv(predictionfilename, index=True)




