import pandas as pd

# Charger le fichier CSV dans un DataFrame
dataframe = pd.read_csv('data/test_data/Battery_RUL_test1.csv')

#Définir un chemin de sortie des données clean
outputcleandata='data/test_data/Battery_RUItest1_sortie.csv'

#Tri des données
def clean_data(dataframe,outputcleandata):
    #Lire chaque ligne du DataFrame
    for index, ligne in dataframe.iterrows():
        
        #Déclaration des variables
        
        #Déclaration des variables des colones
        #Cycle Index
        CI_act=dataframe.loc[index, 'Cycle_Index']
        CI_sup=CI_act+1
        CI_ant=CI_act-1
        #Discharge Time
        DT_act=dataframe.loc[index, 'Discharge Time (s)']
        DT_sup=DT_act
        DT_ant=DT_act
        #Decrement 3.6-3.4V
        Dec64_act=dataframe.loc[index, 'Decrement 3.6-3.4V (s)']
        Dec64_sup=Dec64_act
        Dec64_ant=Dec64_act
        #MaxVoltDischar
        Max_act=dataframe.loc[index, 'Max. Voltage Dischar. (V)']
        Max_sup=Max_act
        Max_ant=Max_act
        #MinVoltChar
        Min_act=dataframe.loc[index, 'Min. Voltage Charg. (V)']
        Min_sup=Min_act
        Min_ant=Min_act
        #Time4.15V
        Time415_act=dataframe.loc[index, 'Time at 4.15V (s)']
        Time415_sup=Time415_act
        Time415_ant=Time415_act
        #TimeConstantCurrent
        TCC_act=dataframe.loc[index, 'Time constant current (s)']
        TCC_sup=TCC_act-200
        TCC_ant=TCC_act+200
        #ChargingTime
        CT_act=dataframe.loc[index, 'Charging time (s)']
        CT_sup=CT_act
        CT_ant=CT_act
        #RUL
        RUL_act=dataframe.loc[index, 'RUL']
        RUL_sup=RUL_act-1
        RUL_ant=RUL_act+1
        if(index!=0):
            #Cycle Index
            CI_ant=dataframe.loc[index-1, 'Cycle_Index']
            #Discharge Time
            DT_ant=dataframe.loc[index-1, 'Discharge Time (s)']
            #Decrement 3.6-3.4V
            Dec64_ant=dataframe.loc[index-1, 'Decrement 3.6-3.4V (s)']
            #MaxVoltDischar
            Max_ant=dataframe.loc[index-1, 'Max. Voltage Dischar. (V)']
            #MinVoltChar
            Min_ant=dataframe.loc[index-1, 'Min. Voltage Charg. (V)']
            #Time4.15V
            Time415_ant=dataframe.loc[index-1, 'Time at 4.15V (s)']
            #TimeConstantCurrent
            TCC_ant=dataframe.loc[index-1, 'Time constant current (s)']
            #ChargingTime
            CT_ant=dataframe.loc[index-1, 'Charging time (s)']
            #RUL
            RUL_ant=dataframe.loc[index-1, 'RUL']
        if(index!=1075):
            #Cycle Index
            CI_sup=dataframe.loc[index+1, 'Cycle_Index']
            #Discharge Time
            DT_sup=dataframe.loc[index+1, 'Discharge Time (s)']
            #Decrement 3.6-3.4V
            Dec64_sup=dataframe.loc[index+1, 'Decrement 3.6-3.4V (s)']
            #MaxVoltDischar
            Max_sup=dataframe.loc[index+1, 'Max. Voltage Dischar. (V)']
            #MinVoltChar
            Min_sup=dataframe.loc[index+1, 'Min. Voltage Charg. (V)']
            #Time4.15V
            Time415_sup=dataframe.loc[index+1, 'Time at 4.15V (s)']
            #TimeConstantCurrent
            TCC_sup=dataframe.loc[index+1, 'Time constant current (s)']
            #ChargingTime
            CT_sup=dataframe.loc[index+1, 'Charging time (s)']
            #RUL
            RUL_sup=dataframe.loc[index+1, 'RUL']  
            
        #Déclaration des variables des vérifs
        #Vérif Decrement 3.6-3.4V'
        Up64=1300
        Down64=100
        #Vérif Charging time
        MarginPhyCT=25/100
        FunctionCTrule=10000*(4.3-Min_act)

        #Vérif Time constant current and Time 4.15V 
        MarginPhyTCC=50/100
        MarginPhyT415=50/100
        #Vérif Decharging time
        MarginPhyDT=25/100
        FunctionDTrule=10000*(4.3-Min_act)
            
        #Init Variables de modifications
        New_CI=CI_act
        New_DT=DT_act
        New_Dec64=Dec64_act
        New_Max=Max_act
        New_Min=Min_act
        New_Time415=Time415_act
        New_TCC=TCC_act
        New_CT=CT_act
        New_RUL=RUL_act
        
        #Vérif 1 - Concerne la relation CI-RUL
        if(CI_act > CI_ant):
            if(RUL_act > RUL_ant):
                New_RUL=1113-CI_act
                
        #Vérif 2 - Concerne le decrement 3.6-3.4V
        if(Up64>Dec64_act>Down64):
            #Validation Dec64
            New_Dec64=Dec64_act
        else:
            #Correction Dec64
            New_Dec64=(Dec64_ant+Dec64_sup)/2
            Dec64_act=New_Dec64
        
        #Vérif Charging time - On pose A = RUL * (4.3 - Min voltage) = MarginPhy - On peut supposer que 10% peut être modifier
        #Si A + 10%*A > Charging time > A - 10%*A OK sinon Charging time n = A
        #On veut pouvoir faire évoluer LinearCTrule pour être de plus en plus précis dans la correction des données
        if(FunctionCTrule*(1+MarginPhyCT)>CT_act>FunctionCTrule*(1-MarginPhyCT)):
            #Validation CT
            New_CT=CT_act
        else:
            #Correction CT
            New_CT=FunctionCTrule
            CT_act=New_CT
        
        #Vérif Time constant current and Time 4.15V 
        #La vérification concerne la validité de la corrélation entre le temps à 4.15V et le temps avec courant constant.
        #Si ls valeurs sont incompatibles, elles sont remplacées en fonction des paramètres inférieurs et suppérieurs.
        if(Time415_act+Dec64_act+Dec64_act*MarginPhyT415>TCC_act>Time415_act+Dec64_act-Dec64_act*MarginPhyT415 and TCC_act<10000):
            #Validation TCC et Time415
            New_TCC=TCC_act
            New_Time415=Time415_act
        else:
            #Correction TCC
            New_TCC=(TCC_ant+TCC_sup)/2
            TCC_act=New_TCC
            if(TCC_act-1000+1000*MarginPhyTCC>Time415_act>TCC_act-1000-1000*MarginPhyTCC):
                New_Time415=Time415_act
            else:
                #Correction Time415
                New_Time415=(Time415_sup+Time415_ant)/2
                Time415_act=New_Time415
                
        #Vérif Discharge time - De même que pour la Vérif Charging time, cette vérif a pour but de supprimer les valeurs abérantes
        #en suivant une loi idéale qui peut être mise a jour par le modèle - FunctionDTrule
        #Dépend du Dec64 et Time415
        
        if(CT_act>DT_act and DT_act>800):
            #Validation DT
            New_DT=DT_act
        else:
            #Correction DT
            New_DT=(DT_ant+DT_sup)/2
            
        
        if(index==0):
            print('Ligne',index+1,': Cycle_Index|Discharge Time (s)|Decrement 3.6-3.4V (s)|Max. Voltage Dischar. (V)|Min. Voltage Charg. (V)|Time at 4.15V (s)|Time constant current (s)|Charging time (s)|RUL')
        print('Ligne',index+2,': CI:',New_CI,' DT:',New_DT,' Dec64:',New_Dec64,' Max:',New_Max,' Min:',New_Min,' Time415:',New_Time415,' TCC:',New_TCC,' CT:',New_CT,' RUL:',New_RUL)
        dataframe.loc[index, 'Cycle_Index'] = New_CI
        dataframe.loc[index, 'Discharge Time (s)'] = New_DT
        dataframe.loc[index, 'Decrement 3.6-3.4V (s)'] = New_Dec64
        dataframe.loc[index, 'Max. Voltage Dischar. (V)'] = New_Max
        dataframe.loc[index, 'Min. Voltage Charg. (V)'] = New_Min
        dataframe.loc[index, 'Time at 4.15V (s)'] = New_Time415
        dataframe.loc[index, 'Time constant current (s)'] = New_TCC
        dataframe.loc[index, 'Charging time (s)'] = New_CT
        dataframe.loc[index, 'RUL'] = New_RUL
        

        # Écrire le DataFrame modifié dans un nouveau fichier CSV
        dataframe.to_csv(outputcleandata, index=False)

clean_data(dataframe,outputcleandata)

