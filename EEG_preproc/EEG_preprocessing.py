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
#%matplotlib qt5


#%%
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
    raw = mne.io.read_raw_edf(file_path)
    raw.info['ch_names']
    sfreq = raw.info['sfreq']
    clean_str = ['EEG','-REF','-AVE','-AV_19','-AV','_av',' '] #for 19ch
    raw, ele_loc = eptool.prep_set_montages(raw, 'standard_1020', clean_str=clean_str,drop_channel='clin_19', plot=False)#19ch

    if os.path.exists(mark_file):
        events = eptool.prep_anywave_markfile(mark_file,sfreq)
        events.columns = ['onset', 'duration', 'description']
        raw.set_annotations(mne.Annotations(onset=events['onset'],
                                            duration=events['duration'], 
                                            description=events['description']))
    else:
        events = raw.annotations.to_data_frame(time_format = 'ms')
        events['onset'] = events['onset']/1000
    #拼接并保存fif文件
    desired_event_ids = ['1','2','3','纺锤节律'] # 需要拼接的事件ID
    for desired_event_id in desired_event_ids:
        selected_segments = []
        desired_events = events[events['description'] == desired_event_id]
        for index, event in desired_events.iterrows():
            start_sample = event['onset'] * sfreq  # 转换为采样点
            stop_sample = start_sample + event['duration'] * sfreq  # 计算结束采样点
            if start_sample < 0 or stop_sample > len(raw.times) or stop_sample <= start_sample:
                print(f"事件 {desired_event_id} 的时间超出范围，跳过该事件")
                continue
            else:
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
            new_raw.save(save_dir, overwrite=True)
            print(f"拼接后的EEG数据已保存为 {save_dir}")
        else:
            print(f"未发现标记为 {desired_event_id}的选段")

#%% -----拼接多个edf文件-----

data_dir = r"H:\msy\小论文_DOC_SBM_EEG\data_00_EEGraw\脑干慢波EDF"
output_dir = r"H:\msy\小论文_DOC_SBM_EEG\data_00_EEGraw"

files = os.listdir(data_dir)
trans_table = str.maketrans('', '', '0123456789')
patient_prefixs_d = [s.translate(trans_table)[0:-4] for s in files]
patient_prefixs = list(set(patient_prefixs_d))

for patient_prefix in patient_prefixs:
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(patient_prefix) and f.endswith('.edf')]
    file_list.sort(key=lambda x: int(os.path.basename(x).split(patient_prefix)[1].split('.edf')[0]))
    raw_comb = eptool.concatenate_edf_files(file_list)
    raw_combb = eptool.mark_bad_around_labels(raw_comb.copy(), target_labels=['BAD boundary'], buffer_sec=0.5, bad_label='bad')
    raw_combb.plot()
    raw_combb.compute_psd().plot()
    raw_combb.export(os.path.join(output_dir, patient_prefix+'.edf'), overwrite=True)
# %%
