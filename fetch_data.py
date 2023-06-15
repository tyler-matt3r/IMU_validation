import boto3
import json
import datetime
import pandas as pd
from io import BytesIO
import correct_drift

CANSERVER_PARSED_BUCKET = 'matt3r-canserver-us-west-2'
CANSERVER_EVENT_BUCKET = 'matt3r-canserver-event-us-west-2'
IMU_BUCKET = 'matt3r-imu-us-west-2'
s3_client = boto3.client('s3')

def get_imu_data(k3y_id, org_id, start_date, end_date):
    return correct_drift.correct_clock(k3y_id, org_id, start_date, end_date)

def get_events(k3y_id, org_id, start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
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

def get_can_data(k3y_id, org_id, start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
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