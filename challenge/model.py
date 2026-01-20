import numpy as np
import pandas as pd

from datetime import datetime
from typing import Tuple, Union, List

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

class FeatureGeneration:
    
    def __init__(self, data: pd.DataFrame):
        self.data             = data.copy()
        self._airlines        = None
        self._top_10_features = None
        
    def _get_period_day(self, date:str) -> str: 
        date_time     = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min   = datetime.strptime("05:00", '%H:%M').time()
        morning_max   = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min   = datetime.strptime("19:00", '%H:%M').time()
        evening_max   = datetime.strptime("23:59", '%H:%M').time()
        night_min     = datetime.strptime("00:00", '%H:%M').time()
        night_max     = datetime.strptime("4:59", '%H:%M').time()
        
        if(date_time >= morning_min and date_time <= morning_max):
            return 'mañana'
        elif(date_time >= afternoon_min and date_time <= afternoon_max):
            return 'tarde'
        elif(
            (date_time >= evening_min and date_time <= evening_max) or
            (date_time >= night_min and date_time <= night_max)
        ):
            return 'noche'
        
    def _is_high_season(self, fecha:str) -> int:
        fecha_año  = int(fecha.split('-')[0])
        fecha      = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
        
    def _get_min_diff(self, data:pd.DataFrame) -> float:
        fecha_o  = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i  = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
        
    def _delay(self, data:pd.DataFrame)-> List[int]:
        threshold_in_minutes = 15
        return np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
    def generate_all(self) -> pd.DataFrame:
        self.data['period_day']  = self.data['Fecha-I'].apply(self._get_period_day)
        self.data['high_season'] = self.data['Fecha-I'].apply(self._is_high_season)
        self.data['min_diff']    = self.data.apply(self._get_min_diff, axis = 1)
        return self.data
        
    def get_features(self)-> pd.DataFrame:
        self.data = self.generate_all()
        features = pd.concat([ 
                                pd.get_dummies(self.data['OPERA'], prefix = 'OPERA'),
                                pd.get_dummies(self.data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                                pd.get_dummies(self.data['MES'], prefix = 'MES')], 
                                axis = 1
                            )
        self._airlines = self.data['OPERA'].unique()
        ### selection Feature Importance
        self._top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        features_importance = features[self._top_10_features]
        return features_importance
        
class DelayModel:

    def __init__(
        self
    ):
        self._model          = LogisticRegression()
        self.top_10_features = None
        self.airlines        = None
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        feaGen   = FeatureGeneration(data)
        features = feaGen.get_features()
        self.airlines        = feaGen._airlines
        self.top_10_features = feaGen._top_10_features
        
        if target_column is not None:
            feaGen.data['delay'] = feaGen._delay(feaGen.data)
            target               = pd.DataFrame(feaGen.data['delay'], columns=['delay'])
            return features, target 
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # factor of balance
        n_y0 = len(target['delay'][target['delay'] == 0])
        n_y1 = len(target['delay'][target['delay'] == 1])
        # model training
        self._model = LogisticRegression(class_weight={1: n_y0/len(target['delay']), 0: n_y1/len(target['delay'])})
        self._model.fit(features, target['delay'])
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        try:
            preds = self._model.predict(features)
        except NotFittedError:
            from pathlib import Path
            # fit with base dataset 
            DATA_PATH = Path(__file__).resolve().parents[1]/"data"/"data.csv"
            data = pd.read_csv(filepath_or_buffer=DATA_PATH)


            X_train, y_train = self.preprocess(data=data, target_column="delay")
            self.fit(features=X_train, target=y_train)
        return self._model.predict(features).astype(int).tolist()
