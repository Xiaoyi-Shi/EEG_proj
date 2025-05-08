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


if __file__ == '__main__':
    import pandas as pd

    import numpy as np
    file_path = r"H:\msy\DOC_eeg\raw_data\other\陈爱珍.edf.mrk"
    makers = pd.read_csv(file_path, sep='\s+', header=None, encoding='gbk')