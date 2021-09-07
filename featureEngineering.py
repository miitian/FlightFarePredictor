import pandas as pd
import numpy as np

# Function to convert duration into minutes

def durationInMin(Duration):
    h = Duration.str.extract('(\d+)h', expand=False).astype(float)*60
    m = Duration.str.extract('(\d+)m', expand=False).astype(float)
    duration_min = h.fillna(0)+m.fillna(0)
    return duration_min


# Function to identify whether departure time during peak hours or not
# morning 9-12 and evening 7-10 are peak hours

def isPeakHourDept(dep_hr):
    return dep_hr.apply(lambda x: 1 if x in([9,10,11,12,19,20,21,22]) else 0)


# Function to identify whether week day is weekend or not
# weekday 4,5,6 are weekend
# weekday 0,1,2,3 are not weekend

def isWeekend(weekDay):
    return weekDay.apply(lambda x: 1 if x in([4,5,6]) else 0)


def fetEngineering(df):
    
    df.Date_of_Journey = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y")
    
    # drop null observations
    df.dropna(inplace=True)
    
    # last four airlines are having very minimal records
    # Either we can drop them or combine them to make premium economy airlines
    # Jet Airways Business has only 6 observations but price for these flights is highest - keep it
    # Vistara Premium economy has only 3 observations and price is almost similar to Vistara flights. We can drop these observations.
    # Trujet has only 1 observations, drop this record
    df.drop(df[df.Airline.isin(["Vistara Premium economy", "Trujet"])].index, axis=0, inplace=True)

    # extract day, weekday and month from Date_of_Journey
    df['flight_day'] = df.Date_of_Journey.dt.day
    df['flight_weekday'] = df.Date_of_Journey.dt.weekday
    df['flight_month'] = df.Date_of_Journey.dt.month

    # extract hour and minutes from Dep_Time
    df['dep_hr'] = pd.to_datetime(df.Dep_Time).dt.hour
    df['dep_minute'] = pd.to_datetime(df.Dep_Time).dt.minute
    #list(map(int,train_df.iloc[1].Dep_Time.split(':', )))

    # apply functions on dataframe
        #1. duration_min
        #2. isWeekend
        #3. isPeakHourDept
    df['duration_min'] = durationInMin(df['Duration'])
    df['is_weekend'] = isWeekend(df['flight_weekday'])
    df['is_peakHourDept'] = isPeakHourDept(df['dep_hr'])

    # Onehot encoding on Airline, Source, Destination columns
    oh_cols = ['Airline', 'Source', 'Destination']    
    oh_df = pd.get_dummies(df, columns=oh_cols, drop_first=True)
    
    # Label encode Total_Stops feature
    df['stops'] = df['Total_Stops'].map({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4})
    
    # Drop unwanted columns 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info'
    drop_cols = ['Date_of_Journey', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info', 'Total_Stops']
    oh_df.drop(drop_cols, axis=1, inplace=True)
    
    return oh_df