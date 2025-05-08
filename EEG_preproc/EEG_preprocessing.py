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

#%% -----128设置montages-----

raw = mne.io.read_raw_bdf(file_path)

###古早128通道名称错乱需要修改
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
}
raw, ele_loc = eptool.prep_set_montages(raw, 'standard_1005',channel_mapping = channel_mapping_128 ,drop_channel='diff')

###新128不需要修改，但是需要把OZ改为Oz
raw, ele_loc = eptool.prep_set_montages(raw, 'standard_1005',channel_mapping = {'OZ':'Oz'} ,drop_channel='diff')

###19通道名称有多余字符需要去除
clean_str = ['EEG','-REF','-AVE','-AV_19','-AV','_av',' ']
raw, ele_loc = eptool.prep_set_montages(raw, 'standard_1020', clean_str=clean_str, drop_channel='clin_19',plot=False)

#%% -----预处理-----

###流程1：加载数据->降采样->滤波->去除工频干扰->设置参考（投影）->初步标记坏导并裁剪干净段落->查看选段->再次判断坏导（根据功率谱）
raw.load_data()                                     # 加载数据
raw.resample(250, npad='auto')                      # 降采样到250HZ
raw.notch_filter(freqs=50)                          # 凹陷滤波50HZ去除市电干扰
raw.filter(1,70)                                    # 滤波，滤过括号范围外的频率
raw.set_eeg_reference(ref_channels='average', projection=True) #设置平均导联投影
raw.plot(n_channels = len(raw.ch_names), butterfly = False)     # 初步标记坏导, 裁剪干净选段
raw_cut = raw.copy().crop(200,500)              # 选段，单位秒
raw_cut_psd = raw_cut.compute_psd(fmax=125,reject_by_annotation=True,proj=True)#查看0-125HZ功率谱
raw_cut_psd.plot(sphere = 'eeglab') 
#raw_cut_psd.plot_topo(dB=True)
raw_cut.plot()                                          # 再次标记坏导,查看选段

###流程2：

#%%-----ICA-----

###流程1：对去除坏导的选段进行ICA->插值坏导->csd重建（可选）->保存数据
raw_ica = eptool.prep_autoICA(raw_cut,n_components=20)
raw_ica.interpolate_bads()
raw_ica.plot()
#raw_csd= eptool.prep_CSD(raw_ica)
raw_ica.save(os.path.join(main_dir, 'ica_raw.fif'), overwrite=True) #保存数据
#raw_csd.save(os.path.join(main_dir, 'csd_raw.fif'), overwrite=True) #保存数据

#%%-----分段-----

###对选段进行分段->查看分段功率谱->查看分段平均波形
raw_seg = mne.make_fixed_length_epochs(raw_ica, duration=5, preload=True, proj=True, overlap=2.5)
raw_seg.plot()
raw_seg.plot_image()

seg_spectrum = raw_seg.compute_psd()
seg_spectrum.plot()
seg_spectrum.plot_topomap( agg_fun=np.median)

raw_seg_avg = raw_seg.average()
raw_seg_avg.plot(gfp=True)
raw_seg_avg.plot_image()

###用基线计算协方差矩阵 
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

#%%-----maker处理-----

###从外置marker文件中读取选段
main_dir = '意识障碍EDF标记'
patient_list = [p for p in os.listdir(main_dir) if p.endswith('edf')]
for i in patient_list:
    file_path = os.path.join(main_dir,i)
    mark_file = file_path +'.mrk'
    if os.path.exists(mark_file):
        raw = mne.io.read_raw_edf(file_path)
        raw.info['ch_names']
        sfreq = raw.info['sfreq']
        clean_str = [' ','EEG','-REF','-AV','_av'] #for 19ch
        raw, ele_loc = eptool.prep_set_montages(raw, 'standard_1020', clean_str=clean_str,drop_channel='clin_19', plot=False)#19ch

        with open(mark_file, 'r', encoding='gbk') as f:
            lines = f.readlines()
        onset = []
        duration = []
        description = []
        for line in lines[1:]:
            description.append(line.split('\t')[0])
            onset.append(line.split('\t')[2])
            duration.append(line.split('\t')[3].split('\n')[0])

        event_dict = {np.str_('1'): 1,
                    np.str_('2'): 2,
                    np.str_('3'): 3}

        valid_values = ['1', '2', '3']
        description = [item.replace('枕区节律','1').replace('背景节律','2').replace('纺锤节律','3') for item in description]
        events = np.array([onset, duration, description]).T
        events[:, 0] = events[:, 0].astype(np.double) * sfreq
        events[:, 1] = events[:, 1].astype(np.double) * sfreq
        events = events[np.isin(events[:, 2], valid_values)].astype(np.double).astype(np.int64)
        events = events[events[:, 1] != 0]

        saved_files = []
        for desired_event_id in range(1,4):
            selected_segments = []
            for event in events:
                onset, duration, event_value = event
                if event_value == event_dict[str(desired_event_id)]:
                    start_sample = onset
                    stop_sample = onset + duration
                    selected_segments.append(raw[:, start_sample:stop_sample][0])
            
            # 4. 合并选定的数据片段
            if selected_segments:
                concatenated_data = np.concatenate(selected_segments, axis=1)
                info = raw.info  # 保留原始数据信息
                new_raw = mne.io.RawArray(concatenated_data, info)
            
                # 5. 保存为新的FIF文件
                if not os.path.exists(os.path.join(file_path.split('.')[0])):
                    os.makedirs(os.path.join(file_path.split('.')[0]))
                save_dir = os.path.join(file_path.split('.')[0] ,'all_'+str(desired_event_id)+'.fif')
                saved_files.append(save_dir)
                new_raw.save(save_dir, overwrite=True)
                print(f"拼接后的EEG数据已保存为 {save_dir}")
            else:
                print(f"未发现标记为 {desired_event_id}的选段")
'''
    #写入raw
    onsets = events[:, 0] / sfreq
    durations = events[:, 1] / sfreq
    descriptions = events[:, 2].astype(str)

    annotations = mne.Annotations(onset=onsets,
                                duration=durations,
                                description=descriptions)
    raw.set_annotations(annotations)
'''
















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
