import pickle 
from model import Input
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from numpy import array as numpy_array, ndarray

class Predictor(): 
    def __init__(self): 
        with open("model.pkl", 'rb') as model_pkl, open("scaler.pkl", 'rb') as scaler_pkl: 
            self.__model_pkl_load: RandomForestClassifier = pickle.load(model_pkl)
            #self.__model_pkl_load = pickle.load(model_pkl)
            print(self.__model_pkl_load)
            
            self.__scaler_pkl_load: StandardScaler = pickle.load(scaler_pkl) 
            #self.__scaler_pkl_load = pickle.load(scaler_pkl) 
            print(self.__scaler_pkl_load)
            
    def predict(self, data: list): 
        data_to_list: ndarray = numpy_array(list(data)).reshape(1, -1)   
        #data_to_list = data  
        data_scaler_convert = self.__scaler_pkl_load.transform(data_to_list) 
        result = self.__model_pkl_load.predict(data_scaler_convert)
        return result

data_predictor = Predictor() 