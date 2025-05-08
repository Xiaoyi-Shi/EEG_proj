### -*- coding: utf-8 -*-
# @Time    : 2025/05/07

import os 
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import EEG_cal_tools as ecal

#%% 计算指定频段功率大小、峰值功率和对称性指数

main_dir = '脑干慢波fif'
bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,12),'beta':(12,20),'gamma':(20,45)}
cal_res_list = []
patient_list = [p for p in os.listdir(main_dir) if p.endswith('.fif')]
for i in patient_list:
    raw = mne.io.read_raw_fif(os.path.join(main_dir,i),preload = True)
    ### 如果未预处理可以在这里进行预处理
    #raw = raw.set_montage('standard_1020')
    #raw.resample(250, npad='auto')                      # 降采样到250HZ
    #raw.notch_filter(freqs=50)                          # 凹陷滤波50HZ去除市电干扰
    #raw.filter(1,45)                                    # 滤波，滤过括号范围外的频率
    #raw.set_eeg_reference(ref_channels='average')
    #raw.crop(tmin=1,tmax=raw.times[-1]-1) # 选段
    cal_res_unstack = []
    cal_res_title = []
    for band_name, (fmin, fmax) in bands.items():
        cal_res = ecal.cal_power(raw ,method='multitaper',fmin=fmin,fmax=fmax)
        cal_res_unstack += cal_res['total_power'].tolist() + cal_res['peak_freq'].tolist() + cal_res['bsi'].tolist()
        cal_res_unstack.append(cal_res['bsi_mean'])
        cal_res_title_temp =[k + '_total_power' for k in cal_res['channel_names']] + \
                            [k + '_peak_freq' for k in cal_res['channel_names']]+ \
                            [k + '_bsi' for k in ['FP1-2', 'F3-4', 'C3-4', 'P3-4', 'O1-2', 'F7-8', 'T3-4', 'T5-6']]
        cal_res_title_temp.append('bsi_mean')
        cal_res_title += [k + '_' + band_name for k in cal_res_title_temp]
    cal_res_dic = dict(zip(cal_res_title, cal_res_unstack))
    cal_res_dic['patient'] = i
    cal_res_list.append(cal_res_dic)
cal_res_list_df = pd.DataFrame(cal_res_list)
cal_res_list_df.to_excel(os.path.join(main_dir,'cal_res_list_df.xlsx') ,index=False)