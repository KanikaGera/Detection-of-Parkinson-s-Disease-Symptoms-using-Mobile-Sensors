import boto3
import decimal
import json
import numpy as np
from numpy import mean, absolute 
from scipy import stats
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import fftpack
import pickle
import math
import random 


class DecimalEncoder(json.JSONEncoder):
   def default(self, o):
       if isinstance(o, decimal.Decimal):
           if o % 1 > 0:
               return float(o)
           else:
               return int(o)
       return super(DecimalEncoder, self).default(o)

def lambda_handler(event, context):
    # TODO implement
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('AccData')
    
    raw_data = table.scan()
    # a=json.dumps(raw_data, cls=DecimalEncoder)
    # print(a)
    data= json.loads(json.dumps(raw_data, cls=DecimalEncoder)) 
    # print(type(data))
    values= data["Items"]
    
    accX=[]
    accY=[]
    accZ=[]
    
    filename= 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    
    for i in range(len(values)):
        accX.append(float(values[i]['accX']))
        accY.append(float(values[i]['accY']))
        accZ.append(float(values[i]['accZ']))
    
    npx= np.asarray(accX)
    npy= np.asarray(accY)
    npz= np.asarray(accZ)
    
    npx_1= np.abs(fftpack.fft(npx))
    npy_1= np.abs(fftpack.fft(npy))
    npz_1= np.abs(fftpack.fft(npz))
    
    def sma(data):
        return np.sum(data)/len(data)

    def energy(data):
        return np.sum(data**2)/len(data)

    def entropy(signal):
        eps=0.00000001
        s= energy(signal)
        entropy = -np.sum(s * np.log2(s + eps))
        return entropy
    
    accX_mean= np.mean(npx_1)
    accY_mean= np.mean(npy_1)
    accZ_mean= np.mean(npz_1)
    accX_mad= mean(absolute(npx_1 - mean(npx_1))) 
    accY_mad= mean(absolute(npy_1 - mean(npy_1))) 
    accZ_mad= mean(absolute(npz_1 - mean(npz_1))) 
    accX_std= np.nanstd(npx_1)
    accY_std= np.nanstd(npy_1)
    accZ_std= np.nanstd(npz_1)
    accX_min= np.min(npx_1)
    accY_min= np.min(npy_1)
    accZ_min= np.min(npz_1)
    accX_max= np.max(npx_1)
    accY_max= np.max(npy_1)
    accZ_max= np.max(npx_1)
    accX_iqr= stats.iqr(npx_1)
    accY_iqr= stats.iqr(npy_1)
    accZ_iqr= stats.iqr(npz_1)
    accX_skew= skew(npx)
    accY_skew= skew(npy)
    accZ_skew= skew(npz)
    accX_kurtosis= kurtosis(npx)
    accY_kurtosis= kurtosis(npy)
    accZ_kurtosis= kurtosis(npz)
    accX_sma= sma(npx)
    accY_sma= sma(npy)
    accZ_sma= sma(npz)
    accX_energy= energy(npx)
    accY_energy= energy(npy)
    accZ_energy= energy(npz)
    accX_entropy= entropy(npx)
    accY_entropy= entropy(npy)
    accZ_entropy= entropy(npz)
    
    insert= [accX_mean, accY_mean, accZ_mean, accX_mad, accY_mad,
      accZ_mad, accX_std, accY_std, accZ_std, accX_min, accY_min,
      accZ_min, accX_max, accY_max, accZ_max, accX_iqr, accY_iqr,
        accZ_iqr,accX_skew, accY_skew, accZ_skew, accX_kurtosis,
        accY_kurtosis, accZ_kurtosis, accX_sma, accY_sma, accZ_sma,
        accX_energy, accY_energy, accZ_energy, accX_entropy,
        accY_entropy, accZ_entropy]
    
    insert_tuple= np.asarray(insert)
    x= insert_tuple.reshape(1,-1)
    
    prediction= loaded_model.predict(x)
    # prediction = random.randint(0,1)

    if(prediction==1):
        sol="Tremor"
    else:
        sol="Steady"

    
    return {
        
        'body': sol
        
    }
