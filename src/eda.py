# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:26:41 2025

@author: apissens
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = (pd.read_csv("data/dataset.csv",          # ← add sep=';'
                  sep=';',                     # the only change
                  decimal=',',                 # optional: if decimals use comma
                  engine='python')             # safer for odd text fields
        .assign(ClaimFreq=lambda d: d.ClaimNb / d.Exposure)
)


sns.histplot(df.ClaimFreq, log_scale=(False, True))
plt.show()

