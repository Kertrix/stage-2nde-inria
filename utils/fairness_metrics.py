import pandas as pd
from fairlearn.metrics import (demographic_parity_difference,
                               demographic_parity_ratio,
                               equal_opportunity_ratio,
                               equalized_odds_difference, false_negative_rate)


class FairnessMetrics():
    def __init__(self, y_true, y_pred, sf_data):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sf_data = sf_data
        
    def demographic_parity(self):
        demo_diff = demographic_parity_difference(self.y_true, self.y_pred, sensitive_features=self.sf_data)
        demo_ratio = demographic_parity_ratio(self.y_true, self.y_pred, sensitive_features=self.sf_data)
        
        print("Demographic parity metrics:")
        print("Demographic partity difference:", demo_diff)
        print("Demographic parity ratio:", demo_ratio)
        
        return demo_diff, demo_ratio
    
    def equalized_odds(self):
        equalized_odds_diff = equalized_odds_difference(self.y_true, self.y_pred, sensitive_features=self.sf_data)
        equalized_odds_ratio = equal_opportunity_ratio(self.y_true, self.y_pred, sensitive_features=self.sf_data)
        
        print("Equalized odds metrics:")
        print("Equalized odds difference:", equalized_odds_diff)
        print("Equalized odds ratio:", equalized_odds_ratio)
        
        return equalized_odds_diff, equalized_odds_ratio  
    
    def fnr(self, groups):
        rate = {}
        for g in groups:
            ids = self.sf_data.loc[self.sf_data == g].index
            rate[g] = false_negative_rate(self.y_true.loc[ids], pd.Series(self.y_pred, index=self.y_true.index).loc[ids])
            
        return rate
        
                                                       