import pandas as pd
import numpy as np
import datetime
import boto3
import json
from io import BytesIO

# define constants
STATIONARY_SPEED = 0.5
SPEED_NOISE_WINDOW = 0.5
BUFFER_TIME = 30
CANSERVER_PARSED_BUCKET = 'matt3r-canserver-us-west-2'
CANSERVER_EVENT_BUCKET = 'matt3r-canserver-event-us-west-2'
IMU_BUCKET = 'matt3r-imu-us-west-2'
s3_client = boto3.client('s3')

# collect the CAN Server and IMU data
def get_events(k3y_id, org_id, start_date, end_date):
    # get a list of all json files in the prefix and filter them to within the date range
    response = s3_client.list_objects(Bucket=CANSERVER_EVENT_BUCKET, Prefix=org_id + '/' + 'k3y-' + k3y_id + '/')
    all_keys = [item['Key'] for item in response['Contents']]
    keys = [file for file in all_keys if file.split('.')[-1] == 'json'
            and len(file.split('/')[-1]) == 15
            and datetime.datetime.strptime(file.split('/')[-1].split('.')[0], '%Y-%m-%d') >= start_date
            and datetime.datetime.strptime(file.split('/')[-1].split('.')[0], '%Y-%m-%d') <= end_date]
    keys = sorted(keys, key=lambda x: x.split('/')[-1].split('.')[0])

    # retrieve and combine filtered json files
    event_dict = {}
    for key in keys:
        response = s3_client.get_object(Bucket=CANSERVER_EVENT_BUCKET, Key=key)
        result = json.loads(response["Body"].read().decode())
        for index in result['imu_telematics']:
            if index in event_dict:
                event_dict[index].extend(result['imu_telematics'][index])
            else:
                event_dict[index] = result['imu_telematics'][index]

    return event_dict

# collect the CAN Server acceleration data
def get_can_data(k3y_id, org_id, start_date, end_date):
    # get a list of all parquet files in the prefix and filter them to within the date range
    response = s3_client.list_objects_v2(Bucket=CANSERVER_PARSED_BUCKET, Prefix=org_id + '/' + 'k3y-' + k3y_id + '/')
    all_keys = [item['Key'] for item in response.get('Contents', [])]

    while response['IsTruncated']:
        response = s3_client.list_objects_v2(Bucket=CANSERVER_PARSED_BUCKET, Prefix=org_id + '/' + 'k3y-' + k3y_id + '/', ContinuationToken=response['NextContinuationToken'])
        all_keys.extend([item['Key'] for item in response.get('Contents', [])])

    keys = [file for file in all_keys if file.split('.')[-1] == 'parquet'
            and datetime.datetime.strptime(file.split('/')[-1].split('_')[0], '%Y-%m-%d') >= start_date
            and datetime.datetime.strptime(file.split('/')[-1].split('_')[0], '%Y-%m-%d') <= end_date]
    keys = sorted(keys, key=lambda x: x.split('/')[-1].split('.')[0])

    # retrieve and combine filtered perquet files
    df_list = []
    for key in keys:
        response = s3_client.get_object(Bucket=CANSERVER_PARSED_BUCKET, Key=key)
        buffer = BytesIO(response['Body'].read())
        can_df = pd.read_parquet(buffer, engine='pyarrow')
        df_list.append(can_df)
    can_df = pd.concat(df_list, axis=0, ignore_index=True)

    return can_df

# collect the IMU acceleration data
def fetch_imu_data(imu_k3y_id, organization_id, start_date, end_date):
    # get a list of all parquet files in the prefix and filter them to within the date range
    response = s3_client.list_objects_v2(Bucket=IMU_BUCKET, Prefix=organization_id + '/' + 'k3y-' + imu_k3y_id + '/accel/')
    all_keys = [item['Key'] for item in response.get('Contents', [])]

    while response['IsTruncated']:
        response = s3_client.list_objects_v2(Bucket=IMU_BUCKET, Prefix=organization_id + '/' + 'k3y-' + imu_k3y_id + '/accel/', ContinuationToken=response['NextContinuationToken'])
        all_keys.extend([item['Key'] for item in response.get('Contents', [])])
    
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

# collect the IMU infer data
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

# correct the timestamp for the IMU data
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

# Filter the driving state data based on CAN Server speed
def get_can_driving_data(can_df, imu_df):
    speed_df = can_df[can_df['speed'].notna()].copy()
    speed_df.reset_index(drop=True, inplace=True)
    speed_df['driving'] = abs(speed_df['speed']) > STATIONARY_SPEED
    dr_start_times = speed_df[speed_df['driving'].astype(int).diff() == 1]['timestamp'].to_list()
    dr_end_times = speed_df[speed_df['driving'].astype(int).diff() == -1]['timestamp'].to_list()

    dr_df_states = []
    for i in range(min(len(dr_start_times),len(dr_end_times))):
        # filter out noise
        if dr_end_times[i] - dr_start_times[i] > SPEED_NOISE_WINDOW:
            dr_df_states.append(imu_df[(imu_df['correct_timestamp'] >= dr_start_times[i]) 
                            & (imu_df['correct_timestamp'] <= dr_end_times[i])])
    return pd.concat(dr_df_states, ignore_index=True)

# Filter the driving state data based on the IMU motion states
def get_imu_driving_data(imu_df, time_df):
    time_df['motion_bin'] = time_df['motion_state'].apply(lambda x: x != 'stationary').astype(int)
    dr_start_times = time_df[time_df['motion_bin'].diff() == 1]['system_clock(epoch)'].to_list()
    dr_end_times = time_df[time_df['motion_bin'].diff() == -1]['system_clock(epoch)'].to_list()

    imu_df['driving_state'] = imu_df['correct_timestamp'].apply(
        lambda x: any(dr_start - BUFFER_TIME <= x <= dr_end for dr_start, dr_end in zip(dr_start_times, dr_end_times))
    )

    imu_dr_df = imu_df[imu_df['driving_state']]
    return imu_dr_df

# compute the true positive rate based on the driving state data
def TPR(can_dr_df, imu_dr_df, event_dict):
    dr_start_times = [state['start'] for state in event_dict['driving_state']]
    dr_end_times = [state['end'] for state in event_dict['driving_state']]

    proxy_set = set(imu_dr_df[imu_dr_df['correct_timestamp'].apply(lambda x: any(start <= x <= end for start, end in zip(dr_start_times, dr_end_times)))]['correct_timestamp'].to_list())
    truth_set = set(can_dr_df['correct_timestamp'].to_list())

    return len(truth_set.intersection(proxy_set)) / len(truth_set)

# compute the false positive rate based on the parked state data
def FPR(imu_df, imu_dr_df, event_dict):
    pk_start_times = [state['timestamp'][0] for state in event_dict['parked_state']]
    pk_end_times = [state['timestamp'][1] for state in event_dict['parked_state']]

    proxy_set = set(imu_dr_df[imu_dr_df['correct_timestamp'].apply(lambda x: any(start <= x <= end for start, end in zip(pk_start_times, pk_end_times)))]['correct_timestamp'].to_list())
    truth_set = set(imu_df[imu_df['correct_timestamp'].apply(lambda x: any(start <= x <= end for start, end in zip(pk_start_times, pk_end_times)))]['correct_timestamp'].to_list())

    return len(proxy_set) / len(truth_set)

if __name__ == "__main__":
    # ============================
    # imput k3y data
    # start_date_str = '2023-07-17'
    # end_date_str = start_date_str
    organization_id = 'hamid'
    k3y_id = '17700cf8'
    # ============================

    current_time = '2023-07-19T21:37:00Z'

    prev_day = datetime.datetime.strptime(current_time, '%Y-%m-%dT%H:%M:%SZ') - datetime.timedelta(days=1)
    start_date = datetime.datetime.combine(prev_day, datetime.time.min)
    end_date = datetime.datetime.combine(prev_day, datetime.time.max)

    # convert to datetime objects
    # start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    # end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

    # fetch the data from the S3 buckets
    imu_df = fetch_imu_data(k3y_id, organization_id, start_date, end_date)
    time_df = fetch_time_data(k3y_id, organization_id, start_date, end_date)
    event_dict = get_events(k3y_id, organization_id, start_date, end_date)
    can_df = get_can_data(k3y_id, organization_id, start_date, end_date)

    # correct the imu time
    imu_df = shift_time(imu_df, time_df)
    imu_df['norm_acc'] = np.sqrt(imu_df['lr_acc(m/s^2)']**2 + imu_df['bf_acc(m/s^2)']**2 + imu_df['vert_acc(m/s^2)']**2)

    # filter only driving state data
    can_dr_df = get_can_driving_data(can_df, imu_df)
    imu_dr_df = get_imu_driving_data(imu_df, time_df)

    # compute the validation metrics
    tpr = TPR(can_dr_df, imu_dr_df, event_dict)
    fpr = FPR(imu_df, imu_dr_df, event_dict)

    # print results
    # print(f'Between {start_date_str} and {end_date_str},')
    print(f'TPR is {round(tpr * 100, 1)}')
    print(f'FPR is {round(fpr * 100, 2)}')