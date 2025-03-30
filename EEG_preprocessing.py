# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:45:35 2025

@author: LIONGAME
"""
#%%
import os 
import mne
import matplotlib.pyplot as plt
import numpy as np
import EEG_preprocessing_tools as eptool
%matplotlib qt5

main_dir = r'F:\Temp\HEEG预处理'
patient_list = [p for p in os.listdir(main_dir) if p.endswith('bdf')]
file_path = os.path.join(main_dir, patient_list[0])

#%% 设置montages
raw = mne.io.read_raw_bdf(file_path)
channel_mapping_128 = {
    'O12H': 'OI2H',
    'O11H':'OI1H',
    'PO1O': 'PO10',
    'P1O':'P10',
    'PPO1OH':'PPO10H',
    'PP06H':'PPO6H',
    '1Z':'IZ',
    'CPR':'CP2',
    'OZ':'Oz'
    #Fpz, Oz, T7, T8 is needed by sphere = 'eeglab'
}#手动修改通道名称

raw, ele_loc = eptool.prep_set_montages(raw, 'standard_1005',channel_mapping = {'OZ':'Oz'} ,drop_channel='diff')
raw.info
#%% 预处理
raw.load_data()                                     # 加载数据
raw.resample(250, npad='auto')                      # 降采样到250HZ
raw.filter(1,70)                                    # 滤波，滤过括号范围外的频率
raw.notch_filter(freqs=50)                          # 凹陷滤波50HZ去除市电干扰
raw.set_eeg_reference(ref_channels='average', projection=True) #设置平均导联投影
raw.plot(n_channels = len(raw.ch_names), butterfly = False)     # 标记坏导,查看选段

raw_cut = raw.copy().crop(200,500)
raw_cut_psd = raw_cut.compute_psd(fmax=125,reject_by_annotation=True,proj=True)#查看0-125HZ功率谱
raw_cut_psd.plot(sphere = 'eeglab') 
#raw_cut_psd.plot_topo(dB=True)
raw_cut.plot()                                          # 标记坏导,查看选段

raw_ica = eptool.prep_autoICA(raw_cut,n_components=20)
raw_ica.interpolate_bads()
raw_ica.plot()

raw_ica.save(os.path.join(main_dir, 'ica_raw.fif'), overwrite=True) #保存ica结果
#raw_csd= eptool.prep_CSD(raw_ica)
#%% 分段
raw_seg = mne.make_fixed_length_epochs(raw_csd, duration=5, preload=True, proj=True, overlap=2.5)
raw_seg.plot()
raw_seg.plot_image()

seg_spectrum = raw_seg.compute_psd()
seg_spectrum.plot()
seg_spectrum.plot_topomap( agg_fun=np.median)

"""
#用基线计算协方差矩阵 
method_params = dict(diagonal_fixed=dict(mag=0.01, grad=0.01, eeg=0.01))
noise_covs = mne.cov.compute_covariance(
    raw_seg,
    method="auto",
    return_estimators=True,
    n_jobs=None,
    projs=None,
    rank=None,
    method_params=method_params,
    verbose=True,
)

# With "return_estimator=True" all estimated covariances sorted
# by log-likelihood are returned.
print("Covariance estimates sorted from best to worst")
for c in noise_covs:
    print(f'{c["method"]} : {c["loglik"]}')
"""
raw_seg_avg = raw_seg.average()
raw_seg_avg.plot(gfp=True)
raw_seg_avg.plot_image()















#%% test
raw_5 = eptool.prep_find_eog(raw_3, 'Fp1')
raw_5 = eptool.prep_muscle_detection(raw_3)
raw_4 = eptool.prep_autoICA(raw_3)
raw_5 = eptool.prep_CSD(raw_4)

#%%
raw = mne.io.read_raw_bdf(file_path, preload=True)
raw.rename_channels(channel_mapping) #修改通道名称
mon_clist = [item.upper() for item in montage.ch_names]
raw_clist = [item.upper() for item in raw.info.ch_names]
raw.rename_channels(dict(zip(raw.info.ch_names, raw_clist)))
difference = [item for item in raw_clist if item not in mon_clist]
raw.drop_channels(difference)


raw.set_montage(montage,match_case=False,on_missing='warn')  #设置通道位置 ，mne.channels.get_builtin_montages() 可以查看montage种类
raw.drop_channels(['VEOL', 'VEOU', 'HEOL', 'HEOR'])

raw_last_3 = raw.copy().crop(3400,3599)
ed_data = mne.preprocessing.compute_bridged_electrodes(raw_last_3)
bridged_idx, ed_matrix = ed_data
mne.viz.plot_bridged_electrodes(
    raw.info,
    bridged_idx,
    ed_matrix,
    title="Bridged Electrodes",
    topomap_args=dict(vlim=(None, 5)),
)

threshold_muscle = 5  
annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
    raw_last_3,
    ch_type="eeg",
    threshold=threshold_muscle,
    min_length_good=0.2,
    filter_freq=[110, 140],
)
fig, ax = plt.subplots()
ax.plot(raw_last_3.times, scores_muscle)
ax.axhline(y=threshold_muscle, color="r")
ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")
order = np.arange(144, 164)
raw.set_annotations(annot_muscle)
raw.plot(start=5, duration=20, order=order)

#%%
raw.set_eeg_reference(ref_channels='average')       #2-设置平均参考
#%%
mne.viz.plot_sensors

mne.plot_sensors()
