# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:17:47 2025

@author: LIONGAME
"""
import mne
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prep_set_montages(raw, montage_name, clean_str=None, channel_mapping=None, drop_channel=None, plot=True):
    """setting montages.
    Parameters
    ----------
    - raw : raw-like Object.
    - montage_name : str 
        more you can see by mne.channels.get_builtin_montages().
    - clean_str=None : list
        the str you want to delete in the channel.
    - channel_mapping=None : dict
        change channel names
    - drop_channel=None : list | str
        if drop_channel='diff' drop the channel which not in standard montages
        if drop_channel='clin_19' drop the channel which not in clin_19
    - plot=True : bool
        weather to plot sensors
        
    Returns raw, position
    -------
    The all montages below are allowed. you can call 
    mne.channels.get_builtin_montages() to review:
    'standard_1005',
     'standard_1020',
     'standard_alphabetic',
     'standard_postfixed',
     'standard_prefixed',
     'standard_primed',
     'biosemi16',
     'biosemi32',
     'biosemi64',
     'biosemi128',
     'biosemi160',
     'biosemi256',
     'easycap-M1',
     'easycap-M10',
     'easycap-M43',
     'EGI_256',
     'GSN-HydroCel-32',
     'GSN-HydroCel-64_1.0',
     'GSN-HydroCel-65_1.0',
     'GSN-HydroCel-128',
     'GSN-HydroCel-129',
     'GSN-HydroCel-256',
     'GSN-HydroCel-257',
     'mgh60',
     'mgh70',
     'artinis-octamon',
     'artinis-brite23',
     'brainproducts-RNP-BA-128'
     """
    raw_c = raw.copy()
    montage = mne.channels.make_standard_montage(montage_name)
    
    if clean_str:
        new_ch_name = raw_c.info.ch_names
        for s in clean_str:
            new_ch_name = [item.replace(s,'') for item in new_ch_name]
        raw_c.rename_channels(dict(zip(raw_c.info.ch_names, new_ch_name)))
        
    if channel_mapping:
        raw_c.rename_channels(channel_mapping) #修改通道名称
        
    mon_clist = [item.upper() for item in montage.ch_names]
    difference = [x for x in raw_c.info.ch_names if x.upper() not in mon_clist]
    if difference:
        print(f"{difference} are not in montages, if you wants to remove these channel, setting drop_channel='diff'")
    if drop_channel:
        if drop_channel == 'diff':
            raw_c.drop_channels(difference)
        elif drop_channel == 'clin_19':
            raw_c.pick_channels(['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz'])
        else:
            raw_c.drop_channels(drop_channel)
            print(f"{difference} are removed")
    
    raw_c.set_montage(montage,match_case=False)
    
    if plot:
        try:
            fig, pos = raw_c.plot_sensors(sphere = 'eeglab', show_names=True)
        except:
            print('not suprot sphere = \'eeglab\' ')
        fig, pos = raw_c.plot_sensors(show_names=True)
        fig, pos = raw_c.plot_sensors(kind = '3d', show_names=True)
    else:
        pos = None
    return raw_c, pos

def prep_bridged_electrodes(dir_path, data_type, clean_str=None, channel_mapping=None, drop_channel=None, plot=True, **kwargs):
    """
    Parameters
    ----------
    
    Returns raw, position
    -------
    """
    pass

def prep_CSD(raw):
    
    raw_c = raw.copy().load_data()
    raw_c.set_eeg_reference(ref_channels='average')
    raw_csd = mne.preprocessing.compute_current_source_density(raw_c)
    raw_c.plot(title = 'raw')
    raw_c.compute_psd().plot()
    raw_csd.compute_psd().plot()
    raw_csd.plot(title = 'raw_csd')
    
    return raw_csd
    
def prep_find_eog(raw, eog_channel, thr=None, ploty=True):
    """
    Parameters
    ----------
    raw : raw-like Object.
    eog_channel : str | list
        the channel contain the eog signal.
    thr: float
        senstive.
    ploty : bool
        weather to plot.
        
    Returns raw
    -------
    """
    raw_c = raw.copy().load_data()
    eog_event_id = 777
    eog_events = mne.preprocessing.find_eog_events(raw_c, eog_event_id,ch_name = eog_channel,thresh=thr)
    
    onset = eog_events[:, 0] / raw_c.info['sfreq']  # 转换为秒
    duration = [0] * len(onset)  # 事件持续时间为 0
    description = ['EOG'] * len(onset)  # 为每个事件标记描述
    eog_annotations = mne.Annotations(onset=onset, duration=duration, description=description,orig_time=raw_c.annotations.orig_time)

    raw_c.set_annotations(raw_c.annotations + eog_annotations)
    
    if ploty == True:
        raw_c.plot()
    
    return raw_c

def prep_autoICA(raw, method='picard',n_components=10, fit_params=None, ploty=True):
    """
    Parameters
    ----------
    raw : raw-like Object.
    method='picard' : str
        other options are 'fastica' , 'infomax' and 'picard'.
    n_components=10 : int
    ploty=True : bool
        weather to plot.
        
    Returns raw
    -------
    """
    raw_c = raw.copy().load_data()
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=52,
    )
    
    ica.fit(raw_c)
    muscle_idx_auto, scores = ica.find_bads_muscle(raw_c)
    print(f"Automatically found muscle artifact ICA components: {muscle_idx_auto}")
    ica.exclude = muscle_idx_auto
    
    if ploty == True:
        title = f"ICA decomposition using {method} "
        ica.plot_components(inst=raw_c,title=title)
        ica.plot_sources(raw_c)
        ica.plot_scores(scores)
        plt.show(block=True)
        ica.plot_overlay(raw_c)
    raw_c = ica.apply(raw_c)
    print(f"Removed ICA components: {ica.exclude}")
    
    return raw_c
    
def prep_muscle_detection(raw, thr=3,filter_freq=[110,124]):
    """
    Parameters
    ----------
    raw : raw-like Object.
    thr : int
        
    Returns raw
    -------
    """
    raw_c = raw.copy().load_data()
    raw_c = raw_c.set_eeg_reference(ref_channels='average')
    # The threshold is data dependent, check the optimal threshold by plotting
    # ``scores_muscle``.
    threshold_muscle = thr  # z-score
    # Choose one channel type, if there are axial gradiometers and magnetometers,
    # select magnetometers as they are more sensitive to muscle activity.
    annot_muscle, scores_muscle = mne.preprocessing.annotate_muscle_zscore(
        raw_c,
        ch_type="eeg",
        threshold=threshold_muscle,
        min_length_good=0.2,
        filter_freq=filter_freq,
    )
    
    fig, ax = plt.subplots()
    ax.plot(raw_c.times, scores_muscle)
    ax.axhline(y=threshold_muscle, color="r")
    ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")
    
    raw_c.set_annotations(raw_c.annotations + annot_muscle)
    raw_c.plot(start=5, duration=20)
    
    return raw_c

def prep_anywave_markfile(file_path, sfreq = 1024, mapping=None):
    """
    Parameters
    ----------
    file_path : str
        the path of mark file.
    sfreq : int
        the sampling frequency.
    mapping : dict
        the mapping of annots.
    --------
    Returns makers_fin : dataframe(/s)
        the makers of the mark file.
    """

    makers = pd.read_csv(file_path, sep='\s+', header=None, encoding='gbk').iloc[1:,:]
    if mapping:
        makers.iloc[:,0] = makers.iloc[:,0].map(mapping)
    makers.iloc[:,2] = makers.iloc[:,2].astype(np.double)
    makers.iloc[:,3] = makers.iloc[:,3].astype(np.double)
    makers_fin = makers.iloc[:,[2,3,0]]
    return makers_fin

def concatenate_edf_files(file_paths):
    """
    Concatenates a list of EDF files into a single MNE Raw object.
    
    Parameters:
    -----------
    file_paths : list of str
        List of paths to EDF files to concatenate in order.
    
    Returns:
    --------
    raw : mne.io.Raw
        The concatenated Raw object.
    
    Notes:
    ------
    This function assumes all EDF files have compatible parameters (e.g., same sampling rate and channels).
    It reports key parameters for each file during loading.
    """
    if not file_paths:
        raise ValueError("file_paths list cannot be empty.")
    
    raws = []
    for i, path in enumerate(file_paths, start=1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        print(f"Loading file {i}: {path}")
        raw = mne.io.read_raw_edf(path, preload=True)
        
        # Report key parameters
        print(f"  Sampling frequency: {raw.info['sfreq']:.2f} Hz")
        print(f"  Number of channels: {len(raw.info['ch_names'])} channels: {raw.info['ch_names']}")
        print(f"  Duration: {raw.times[-1]:.2f} seconds")
        print("-" * 50)
        
        raws.append(raw)
    
    # Concatenate all Raw objects
    concatenated_raw = mne.concatenate_raws(raws)
    
    print(f"Concatenation complete. Total duration: {concatenated_raw.times[-1]:.2f} seconds")
    
    return concatenated_raw

def mark_bad_around_labels(raw, target_labels, buffer_sec=1.0, bad_label='BAD'):
    """
    在 raw 对象的 annotations 中，针对指定标签的前后 buffer_sec 秒标记为坏段。
    
    Parameters:
    -----------
    raw : mne.io.Raw
        输入的 Raw 对象，必须包含 annotations。
    
    target_labels : list of str
        要针对的标签列表，例如 ['seizure_start', 'artifact']。
    
    buffer_sec : float, default=1.0
        前后缓冲秒数。
    
    bad_label : str, default='BAD'
        坏段的描述标签（MNE 默认坏段标签）。
    
    Returns:
    --------
    raw : mne.io.Raw
        修改后的 Raw 对象（annotations 已更新）。
    
    Notes:
    ------
    - 该函数会扩展每个匹配标签的 annotation 时间范围（onset - buffer 到 onset + duration + buffer）。
    - 新坏段会合并到现有 annotations 中（使用 + 操作符，MNE 会自动处理重叠）。
    - 如果 raw 没有 annotations，会自动创建。
    - 确保缓冲不超出数据边界（起始时间 >=0）。
    """
    if not hasattr(raw, 'annotations') or raw.annotations is None:
        raw.set_annotations(mne.Annotations([], [], orig_time=raw.info['meas_date']))
    
    ann = raw.annotations
    matching_indices = [i for i, desc in enumerate(ann.description) if desc in target_labels]
    
    if not matching_indices:
        print("未找到匹配的标签，跳过标记。")
        return raw
    
    new_onsets = []
    new_durations = []
    new_descriptions = []
    
    for i in matching_indices:
        onset = ann.onset[i]
        duration = ann.duration[i]
        
        # 计算扩展范围
        start = max(0.0, onset - buffer_sec)
        end = onset + duration + buffer_sec
        new_dur = end - start
        
        new_onsets.append(start)
        new_durations.append(new_dur)
        new_descriptions.append(bad_label)
    
    # 创建新的坏段 annotations
    bad_ann = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=ann.orig_time
    )
    
    # 合并到现有 annotations（MNE 会处理重叠）
    updated_ann = ann + bad_ann
    raw.set_annotations(updated_ann)
    
    print(f"已标记 {len(matching_indices)} 个标签的前后 {buffer_sec} 秒为坏段。")
    print(f"总坏段数量: {sum(1 for d in updated_ann.description if d == bad_label)}")
    
    return raw

if __file__ == '__main__':
    import pandas as pd

    import numpy as np
    file_path = r"H:\msy\DOC_eeg\raw_data\other\陈爱珍.edf.mrk"
    makers = pd.read_csv(file_path, sep='\s+', header=None, encoding='gbk')