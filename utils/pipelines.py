from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class Pipelines():
    def __init__(self, columns):       
        self.columns = columns
        
    def preprocessing(self):
        categorical_transformer = Pipeline(
        [
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_transformer, self.columns)
            ]
        )
        
        return preprocessor

    def regressor(self):
        return Pipeline(
            [
                ("preprocessor", self.preprocessing()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        
    def random_forest(self):
        return Pipeline(
            [
                ("preprocessor", self.preprocessing()),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        
        
    