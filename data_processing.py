import pandas as pd
import numpy as np
from datetime import timedelta
import os

def get_num_patient_before(df):
    count_list = [0]
    for i in range(1, len(df)):

        count = 0
        limit = (df['Arrival Date'][i] - timedelta(minutes=90))

        while i > 0:
            if df['Dr Seen Date'][i - 1] > limit:
                count += 1
                i -= 1

            elif df['Dr Seen Date'][i - 1] <= limit:
                break

        count_list.append(count)

    return count_list


def group_month(month):
    spring = list(np.arange(3, 7))
    summer = list(np.arange(7, 10))
    fall = list(np.arange(10, 11))
    if month in spring:
        return 'spring'
    elif month in summer:
        return 'summer'
    elif month in fall:
        return 'fall'
    else:
        return 'winter'


def group_hour(hour):
    morning = list(np.arange(0, 13))
    afternoon = list(np.arange(13, 17))
    if hour in morning:
        return 'morning'
    elif hour in afternoon:
        return 'afternoon'
    else:
        return 'evening'


def create_df(verbose=True):
    df = pd.read_excel('ED_data.xlsx')
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    df.drop(['Departure Status Desc.',
             'Depart. Dest. Desc.',
             'Presenting Complaint Desc.',
             'Diagnosis Desc.'], axis=1, inplace=True)

    df = df.rename(columns={' Age  (yrs)': 'Age'})
    df['Triage Priority'] = df['Triage Priority'].astype('category')
    df['Depart Status Code'] = df['Depart Status Code'].astype('category')
    df['Depart. Dest. Code'] = df['Depart. Dest. Code'].astype('category')
    df['Presenting Complaint Code'] = df['Presenting Complaint Code'].astype('category')
    df['Diag Code'] = df['Diag Code'].astype('category')

    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Year'] = df['Arrival Date'].apply(lambda x: x.year)
    df['Arrival_month'] = df['Arrival Date'].apply(lambda x: x.month)
    df['Arrival_month'] = df['Arrival_month'].astype('category')

    df['Arrival_day'] = df['Arrival Date'].apply(lambda x: x.day)
    df['Arrival_day'] = df['Arrival_day'].astype('category')

    df['Arrival_hour'] = df['Arrival Date'].apply(lambda x: x.hour)
    df['Arrival_hour'] = df['Arrival_hour'].astype('category')

    df['waiting_time'] = df['TimeDiff Arrival-Actual Depart (mins)'] - df['TimeDiff TreatDrNr-Act. Depart (mins)']

    df = df.sort_values(by='Arrival Date')

    df['Arrival_period'] = df['Arrival_hour'].apply(lambda x: group_hour(x))
    df['Arrival_season'] = df['Arrival_month'].apply(lambda x: group_month(x))

    df['num_patient_before'] = get_num_patient_before(df)

    df = df[['Age','Triage Priority','Arrival_period','Arrival_season',
             'num_patient_before', 'waiting_time']]

    Q25 = df['waiting_time'].quantile(0.25)
    Q75 = df['waiting_time'].quantile(0.75)
    IQR = Q75 - Q25
    df = df[df['waiting_time'] <= 1.5 * IQR]

    feature_dummies = pd.get_dummies(df[['Triage Priority',
                                         'Arrival_period',
                                         'Arrival_season']], drop_first=True)

    df = pd.concat([df[['Age','num_patient_before']],feature_dummies,df['waiting_time']], axis=1)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print('Data imported!')
        print('Mem. usage is {:5.2f} Mb'.format(end_mem))

    return df


if __name__ == "__main__":

    df = create_df()
    df.to_pickle('df_processed.pkl')

    folder_dir = os.path.dirname(os.path.abspath(__file__))
    print('file location: /n')
    print(folder_dir)

