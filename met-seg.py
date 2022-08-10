import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

data = pd.read_csv("dados_estruturados.csv", header = 0)
data.head()

# converting column data to list
time = data['time'].tolist()
average = data['avg'].tolist()
bps = data['bps'].tolist()

# Função que realiza o EMA a partir de um valor de entrada para a janela e período.

def ema(values, period):
    df_test = pd.DataFrame(data = values)
    df_test_ewma = df_test.ewm(span=period).mean()
    return df_test_ewma

def adjusted_model(Twindow, model):
    Twindow = 20
    final = list()
    ema2 = ema(model, Twindow)
    df = pd.DataFrame(ema2).to_numpy()
    for i in range(0, len(model)):
        x = int(i) * int(df[i])
        final.append(x)
    return final


# Pseudocode for predicting network traffic

def predict(Twindow, data, model):
    tmTrace = data
    window = Twindow
    model2 = adjusted_model(Twindow, model)
    vPModel = model2
    vPrediction = list()

    for i in range (0, len(tmTrace), 1):
        nextValue = 0
        for j in range (0, window):
            if i - j >= 0:
                tmp = (vPModel[j] * tmTrace[i - j])
            nextValue = (nextValue + tmp)
        vPrediction.append(nextValue)
    return vPrediction

# Função que faz a análise do threshold dos valores a armazena os alarmes registrados.

def threshold(ori, pred, threshold):
    alarm = list()
    for i in range(0, len(pred)):
        if (pred[i] - ori[i]) > threshold:
            alarm.append(pred[i])
    print("A quantidade de alarmes foi de: ", len(alarm))
    return alarm

# Função responsável por calcular o NMSE.

def NMSE(A, B):
    mse = mean_squared_error(A, B)
    return mse


# Função responsável por calcular o MAPE. 

def MAPE(Y_actual,Y_Predicted):
    Y_actual, Y_Predicted = np.array(Y_actual), np.array(Y_Predicted)
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def media(data): 
    n = len(data)
    media = 0
    for i in data:
        media = media + i
    result = media / n
    return result


# Chamadas de funções para a predição.

# Predição do Average

predict_tcp = predict(20, average, average)

threshold_media_tcp = media(predict_tcp)

threshold_tcp = threshold(average, predict_tcp, threshold_media_tcp)

NMSE2 = NMSE(average, predict_tcp)
print("NMSE", NMSE2)
MAPE2 = MAPE(average, predict_tcp)
print("MAPE", MAPE2, "\n")

# Predição do Bps

pred_bps = predict(20, bps, bps)

threshold_media_n = media(pred_bps)

threshold_n = threshold(bps, pred_bps, threshold_media_n)

NMSE2 = NMSE(bps, pred_bps)
print("NMSE", NMSE2)
MAPE2 = MAPE(bps, pred_bps)
print("MAPE", MAPE2, "\n")
