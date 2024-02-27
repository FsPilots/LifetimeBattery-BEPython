import pandas as pd


# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/Battery_RUL_test1.csv')

# Afficher les premières lignes du DataFrame
print(dataframe.head())

#Trier les données
#Partie 1 Données abérentes
#Vérif 1 - Si Cycle Index n > Cycle Index n-1 et RUL n < RUL n-1 OK sinon RUL n = (RULn-1) - 1  
#Vérif 2 - On pose RUL * (Max voltage - Min voltage) = A - On peut supposer que 10% peut être modifier
#Si RA + 10%*A > Charging time > A - 10%*A OK sinon Charging time n = A
#Vérif 3 - Si Discharge Time 

#Lire chaque ligne du DataFrame
for index, ligne in dataframe.iterrows():
    if(index==0):
        index=index+1
    else:
        #Cycle Index
        CI_ant=dataframe.loc[index-1, 'Cycle_Index']
        CI_act=dataframe.loc[index, 'Cycle_Index']
        #Discharge Time
        DT_ant=dataframe.loc[index-1, 'Discharge Time (s)']
        DT_act=dataframe.loc[index, 'Discharge Time (s)']
        #Decrement 3.6-3.4V
        Dec64_ant=dataframe.loc[index-1, 'Decrement 3.6-3.4V (s)']
        Dec64_act=dataframe.loc[index, 'Decrement 3.6-3.4V (s)']
        #MaxVoltDischar
        Max_ant=dataframe.loc[index-1, 'Max. Voltage Dischar. (V)']
        Max_act=dataframe.loc[index, 'Max. Voltage Dischar. (V)']
        #MinVoltChar
        Min_ant=dataframe.loc[index-1, 'Min. Voltage Charg. (V)']
        Min_act=dataframe.loc[index, 'Min. Voltage Charg. (V)']
        #Time4.15V
        Time415_ant=dataframe.loc[index-1, 'Time at 4.15V (s)']
        TIme415_act=dataframe.loc[index, 'Time at 4.15V (s)']
        #TimeConstantCurrent
        TCC_ant=dataframe.loc[index-1, 'Time constant current (s)']
        TCC_act=dataframe.loc[index, 'Time constant current (s)']
        #ChargingTime
        CT_ant=dataframe.loc[index-1, 'Charging time (s)']
        CT_act=dataframe.loc[index, 'Charging time (s)']
        #RUL
        RUL_ant=dataframe.loc[index-1, 'RUL']
        RUL_act=dataframe.loc[index, 'RUL']
        
        #Variables de modifications
        New_CI=CI_act
        New_DT=DT_act
        New_Dec64=Dec64_act
        New_Max=Max_act
        New_Min=Min_act
        New_Time415=TIme415_act
        New_TCC=TCC_act
        New_CT=CT_act
        New_RUL=RUL_act
        
        #Vérif 1 - Si Cycle Index n > Cycle Index n-1 et RUL n < RUL n-1 OK sinon RUL n = (RULn-1) - 1
        if(CI_act > CI_ant):
            if(RUL_act > RUL_ant):
                New_RUL=RUL_ant-1
        else:
            index=index+1
        
        
        
        print('Nouvelle Ligne:CI:',New_CI,' DT:',New_DT,' Dec64:',New_Dec64,' Max:',New_Max,' Min:',New_Min,' Time415:',New_Time415,' TCC:',New_TCC,' CT:',New_CT,' RUL:',New_RUL)




