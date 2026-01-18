import pandas as pd

from datetime import datetime
from typing import Tuple, Union, List

class FeatureGeneration:
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        
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


class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
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
        return

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
        return