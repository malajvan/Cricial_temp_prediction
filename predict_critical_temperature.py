import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy import stats

#load train set csv data
train_set=pd.read_csv("data/train.csv").dropna(axis=0)
train_set[(np.abs(stats.zscore(train_set))<3).all(axis=1)]

y=train_set.critical_temp
x=train_set.drop(columns=['critical_temp'])






#filter out feature

def test_del(x):
    new_x= x
    tx,vx,ty,vy=train_test_split(new_x,y,test_size=0.3,random_state=1)
    forest1=RandomForestRegressor(max_depth=20,n_estimators=10,random_state=1)
    forest1.fit(tx,ty)
    forest_y=forest1.predict(vx)
    mve=mean_absolute_error(vy,forest_y)
    print(mve)
    for column in x.copy().columns:
        test_x=new_x.drop(columns=[column])
        tx,vx,ty,vy=train_test_split(test_x,y,test_size=0.3,random_state=1)
        forest1=RandomForestRegressor(max_depth=20,n_estimators=10,random_state=1)
        forest1.fit(tx,ty)
        forest_y=forest1.predict(vx)
        mve_test=mean_absolute_error(vy,forest_y)

        if mve>=mve_test:
            new_x=test_x
            mve=mve_test
            print(column,mve)
    return(new_x)

columns=['mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass',

       'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass',
       'wtd_range_atomic_mass', 'std_atomic_mass', 'wtd_std_atomic_mass',
       'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie',
       'wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie',
       'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius',
       'gmean_atomic_radius', 'wtd_gmean_atomic_radius',
       'entropy_atomic_radius', 'wtd_entropy_atomic_radius',
       'range_atomic_radius', 'wtd_range_atomic_radius', 'std_atomic_radius',
       'wtd_std_atomic_radius', 'wtd_mean_Density', 'gmean_Density',
       'wtd_gmean_Density', 'entropy_Density', 'range_Density',
       'wtd_range_Density', 'std_Density', 'wtd_std_Density',
       'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity',
       'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity',
       'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
       'wtd_range_ElectronAffinity', 'std_ElectronAffinity',
       'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat',
       'gmean_FusionHeat', 'wtd_gmean_FusionHeat', 'entropy_FusionHeat',
       'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat',
       'std_FusionHeat', 'wtd_std_FusionHeat', 'wtd_mean_ThermalConductivity',
       'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity',
       'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity',
       'range_ThermalConductivity', 'wtd_range_ThermalConductivity',
       'std_ThermalConductivity', 'wtd_std_ThermalConductivity',
       'mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence', 'entropy_Valence',
       'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence',
       'std_Valence', 'wtd_std_Valence']

test_set=pd.read_csv("data/test.csv").fillna(0).drop(columns=['index'])
forest2=RandomForestRegressor(max_depth=20,n_estimators=10,random_state=1)
forest2.fit(x,y)
prediction=forest2.predict(test_set)
df=pd.DataFrame()
df['critical_temp']=prediction
df.to_csv("submission.csv",index_label='index')
