import pandas as pd
import matplotlib.pyplot as plt
import fetch_data
import numpy as np
import correct_drift
import datetime

ac_sensor = {'name' : 'acc',
             'units' : 'm/s^2',
             'title' : 'Acceleration',
             'scale' : 15}

def triaxis_plot(imu_df, can_df, start_time, sensor=ac_sensor):
    name = sensor['name']
    units = sensor['units']
    title = sensor['title']
    scale = sensor['scale']
    if 'correct_timestamp' in imu_df.columns:
        time_data = 'correct_timestamp'
    else:
        time_data = 'timestamp(epoch in sec)'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    fig.set_size_inches(10,6)

    ax1.scatter(x=can_df['timestamp'] - start_time, y=-can_df[f'lr_{name}'], s=1, color='red', label='CANserver')
    ax1.scatter(x=imu_df[time_data] - start_time, y=imu_df[f'lr_{name}({units})'], s=1, label='IMU')
    ax1.set_ylim(-scale, scale)

    ax2.scatter(x=can_df['timestamp'] - start_time, y=can_df[f'bf_{name}'], s=1, color='red', label='CANserver')
    ax2.scatter(x=imu_df[time_data] - start_time, y=imu_df[f'bf_{name}({units})'], s=1, label='IMU')
    ax2.set_ylim(-scale, scale)

    ax3.scatter(x=can_df['timestamp'] - start_time, y=can_df[f'vert_{name}'], s=1, color='red', label='CANserver')
    ax3.scatter(x=imu_df[time_data] - start_time, y=imu_df[f'vert_{name}({units})'], s=1, label='IMU')
    ax3.set_ylim(-scale, scale)

    fig.suptitle(f"IMU {title} Metrics for Driving State Data")
    fig.text(0.5, 0.0, 'Elapsed Time (s)', ha='center')

    ax1.set_ylabel(f"Left/Right {title} ({units})")
    ax2.set_ylabel(f"Back/Front {title} ({units})")
    ax3.set_ylabel(f"Vertical {title} ({units})")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    plt.show()

def double_plot(x_data, x_data2, y_data1, y_data2, title, write_slope=True):

    fig, ax = plt.subplots()
    ax.scatter(x=x_data, y=y_data1, s=1)
    ax.scatter(x=x_data2, y=y_data2, s=1)
    plt.title(title, wrap = True)
    plt.xlabel('Elapsed Time (h)')
    plt.ylabel('Clock Difference (s)')

    # set gridlines
    ax.grid(True)
    ax.minorticks_on()
    ax.grid(which='major', color='grey', linestyle='-')
    ax.grid(which='minor', color='lightgrey', linestyle=':')
    # ax.set_xlim(left=0)

    if write_slope:
        m, b = np.polyfit(x_data, y_data1, 1)
        plt.text(0.02, 0.95, f'Slope: {m:.3f} s/h', transform=plt.gca().transAxes)

    plt.show()