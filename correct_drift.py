import boto3
import numpy as np
import datetime
import pandas as pd
from io import BytesIO

IMU_BUCKET = 'matt3r-imu-us-west-2'
s3_client = boto3.client('s3')

def fetch_imu_data(imu_k3y_id, organization_id, start_date, end_date):
    # get a list of all parquet files in the prefix and filter them to within the date range
    response = s3_client.list_objects(Bucket=IMU_BUCKET, Prefix=organization_id + '/' + 'k3y-' + imu_k3y_id + '/accel/')
    all_keys = [item['Key'] for item in response['Contents']]
    keys = [file for file in all_keys if file.split('.')[-1] == 'parquet'
            and len(file.split('/')[-1].split('_')[0]) == 10
            and datetime.datetime.strptime(file.split('/')[-1].split('_')[0], '%Y-%m-%d') >= start_date
            and datetime.datetime.strptime(file.split('/')[-1].split('_')[0], '%Y-%m-%d') <= end_date]
    keys = sorted(keys, key=lambda x: x.split('/')[-1].split('.')[0])

    # retrieve and combine filtered perquet files
    df_list = []
    for key in keys:
        response = s3_client.get_object(Bucket=IMU_BUCKET, Key=key)
        buffer = BytesIO(response['Body'].read())
        imu_df = pd.read_parquet(buffer, engine='pyarrow')
        df_list.append(imu_df)
    imu_df = pd.concat(df_list, axis=0, ignore_index=True)

    return imu_df

def fetch_time_data(imu_k3y_id, organization_id, start_date, end_date):
    # create a 1 day buffer to capture any data on the boundaries
    start_date = start_date - datetime.timedelta(days=1)
    end_date = end_date + datetime.timedelta(days=1)
    # get a list of all parquet files in the prefix and filter them to within the date range
    response = s3_client.list_objects(Bucket=IMU_BUCKET, Prefix=organization_id + '/' + 'k3y-' + imu_k3y_id + '/infer/')
    all_keys = [item['Key'] for item in response['Contents']]
    keys = [file for file in all_keys if file.split('.')[-1] == 'parquet'
            and len(file.split('/')[-1].split('.')[0]) != 10
            and datetime.datetime.strptime(file.split('/')[-1].split('.')[0].split('_')[-1], '%Y-%m-%d') >= start_date
            and datetime.datetime.strptime(file.split('/')[-1].split('.')[0].split('_')[-1], '%Y-%m-%d') <= end_date]
    keys = sorted(keys, key=lambda x: x.split('/')[-1].split('.')[0])

    # retrieve and combine filtered perquet files
    df_list = []
    for key in keys:
        response = s3_client.get_object(Bucket=IMU_BUCKET, Key=key)
        buffer = BytesIO(response['Body'].read())
        time_df = pd.read_parquet(buffer, engine='pyarrow')
        df_list.append(time_df)
    time_df = pd.concat(df_list, axis=0, ignore_index=True)

    # drop any nan values
    time_df.dropna(subset=['diff_sw_sys(second)', 'imu_sw_clock(epoch)', 'system_clock(epoch)'], inplace=True)
    time_df.reset_index(drop=True, inplace=True)

    return time_df

def shift_time(imu_df, time_df):
    # identify any jumps in the data
    jump_limit = 2
    jump_indexes = time_df[abs(time_df['diff_sw_sys(second)'].diff()) > jump_limit].index
    jump_indexes = jump_indexes.append(pd.Index([time_df.index[-1]]))

    # create a list of the slope segments
    segments = []
    index_start = 0
    for index in jump_indexes:
        seg_data = {}
        seg_data['start_timestamp'] = time_df['imu_sw_clock(epoch)'].iloc[index_start]
        seg_data['end_timestamp'] = time_df['imu_sw_clock(epoch)'].iloc[index]
        seg_data['slope'], seg_data['intercept'] = np.polyfit(time_df['system_clock(epoch)'][index_start:index], 
                                                            time_df['diff_sw_sys(second)'][index_start:index], 1)
        seg_data['offset'] = seg_data['slope'] * seg_data['start_timestamp'] + seg_data['intercept']
        segments.append(seg_data)
        index_start = index

    for seg in segments:
        imu_df_seg = imu_df[(imu_df['timestamp(epoch in sec)'] >= seg['start_timestamp'])
                            & (imu_df['timestamp(epoch in sec)'] < seg['end_timestamp'])]
        imu_df.loc[imu_df_seg.index, 'correct_timestamp'] = imu_df_seg['timestamp(epoch in sec)'].apply(
            lambda x: x - (x - seg['start_timestamp']) * seg['slope'] - seg['offset'])

    # drop any nan values
    imu_df.dropna(inplace=True)
    imu_df.reset_index(drop=True, inplace=True)

    return imu_df

def correct_clock(imu_k3y_id, organization_id, start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    imu_df = fetch_imu_data(imu_k3y_id, organization_id, start_date, end_date)
    time_df = fetch_time_data(imu_k3y_id, organization_id, start_date, end_date)
    return shift_time(imu_df, time_df)