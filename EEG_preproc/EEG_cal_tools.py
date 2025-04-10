import mne
import os
import pandas as pd
import numpy as np

def cal_power(raw, method='welch', fmin=1, fmax=50, picks=None, verbose=True, **method_kw):
    """
    计算 EEG 各通道的总功率和峰值功率。
    参数:
    ----------
    - raw: mne.io.Raw 对象, 原始 EEG 数据
    - method: str, 功率计算方法, 支持 'welch' 或 'multitaper'
    - fmin: float, 最小频率 (Hz)
    - fmax: float, 最大频率 (Hz)
    - picks: list 或 None, 选择要分析的通道
    - method_kw: dict, n_fft, n_overlap, n_per_seg, average, window for Welch method, 
                       or bandwidth, adaptive, low_bias, normalization for multitaper method
    - n_fft(Welch) : int, The length of FFT used, must be >= n_per_seg (default: 256). The segments will be zero-padded if n_fft > n_per_seg.
    - n_overlap(Welch) : int, The number of points of overlap between segments. Will be adjusted to be <= n_per_seg. The default value is 0
    - n_per_seg(Welch) : int, Length of each Welch segment (windowed with a Hamming window). Defaults to None, which sets n_per_seg equal to n_fft.
    - average(Welch) : str, How to average the segments. 
        If mean (default), calculate the arithmetic mean. 
        If median, calculate the median, corrected for its bias relative to the mean. 
        If None, returns the unaggregated segments
    - bandwidth(multitaper) : float, Frequency bandwidth of the multi-taper window function in Hz. 
        For a given frequency, frequencies at ± bandwidth / 2 are smoothed together. 
        The default value is a bandwidth of 8 * (sfreq / n_times)
    - normalization : 'full' | 'length'
        Normalization strategy. If “full”, the PSD will be normalized by the sampling rate as well as the length of the signal (as in Nitime). Default is 'length'.
    - verbose: bool, 是否打印结果
    
    返回:
    - dict, 包含通道名称、总功率、峰值功率和频率数组
    ----------
    """

    raw_c = raw.copy().apply_proj()
    # 默认选择 EEG 通道
    if picks is None:
        picks = mne.pick_types(raw_c.info, eeg=True, eog=False, stim=False)
    
    psd = raw_c.compute_psd(method = method, fmin=fmin, fmax=fmax, picks=picks, **method_kw)
    psds, freqs = psd.get_data(return_freqs = True)
    
    # 计算总功率和峰值功率
    total_power = psds.sum(axis=1)  # 每个通道的总功率
    peak_power = psds.max(axis=1)   # 每个通道的峰值功率
    peak_freq = freqs[np.argmax(psds, axis=1)]  # 整体峰值频率

    # 获取通道名称
    ch_names = [raw_c.ch_names[i] for i in picks]
    
    # 计算BSI
    left_channels = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']
    right_channels = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    channel_pairs = list(zip(left_channels, right_channels))

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_psds = psds[:, band_mask]
    
    # 提取左右通道功率
    left_pow = []
    right_pow = []
    for left_ch, right_ch in channel_pairs:
        if any(item.lower() == left_ch.lower() for item in ch_names) and any(item.lower() == right_ch.lower() for item in ch_names):
            left_idx = ch_names.index(left_ch)
            right_idx = ch_names.index(right_ch)
            left_pow.append(band_psds[left_idx].mean())
            right_pow.append(band_psds[right_idx].mean())
    
    left_pow = np.array(left_pow)
    right_pow = np.array(right_pow)
    
    # 计算 BSI
    bsi = np.abs(left_pow - right_pow) / (left_pow + right_pow) # 计算 BSI, 数值越小表示左右脑越对称
    bsi_mean = np.mean(bsi)

    # 返回结果
    return {
        'channel_names': ch_names,
        'total_power': total_power,
        'peak_power': peak_power,
        "peak_freq": peak_freq,
        'frequencies': freqs,
        'psds': psds,
        'bsi': bsi,
        'bsi_mean': bsi_mean
    }

if __name__ == "__main__":
        
    main_dir = 'raw_data_clips'
    bands = {'delta':(1,4),'theta':(4,8),'alpha':(8,12),'beta':(12,20),'gamma':(20,45)}
    cal_res_list = []
    for i in os.scandir(main_dir):
        if i.is_dir():
            patient_list = [p for p in os.listdir(i) if p.endswith('.fif')]
            for j in patient_list:
                raw = mne.io.read_raw_fif(os.path.join(i.path,j),preload = True)
                raw = raw.set_montage('standard_1020')
                raw.resample(250, npad='auto')                      # 降采样到250HZ
                raw.notch_filter(freqs=50)                          # 凹陷滤波50HZ去除市电干扰
                raw.filter(1,45)                                    # 滤波，滤过括号范围外的频率
                raw.set_eeg_reference(ref_channels='average')
                cal_res_unstack = []
                cal_res_title = []
                for band_name, (fmin, fmax) in bands.items():
                    cal_res = cal_power(raw ,method='multitaper',fmin=fmin,fmax=fmax)
                    cal_res_unstack += cal_res['total_power'].tolist() + cal_res['peak_freq'].tolist() + cal_res['bsi'].tolist()
                    cal_res_unstack.append(cal_res['bsi_mean'])
                    cal_res_title_temp =[i + '_total_power' for i in cal_res['channel_names']] + \
                                        [i + '_peak_freq' for i in cal_res['channel_names']]+ \
                                        [i + '_bsi' for i in ['FP1-2', 'F3-4', 'C3-4', 'P3-4', 'O1-2', 'F7-8', 'T3-4', 'T5-6']]
                    cal_res_title_temp.append('bsi_mean')
                    cal_res_title += [i + '_' + band_name for i in cal_res_title_temp]
                cal_res_dic = dict(zip(cal_res_title, cal_res_unstack))
                cal_res_dic['patient'] = i.name + '_' + j.split('.')[0]
                cal_res_list.append(cal_res_dic)
    cal_res_list_df = pd.DataFrame(cal_res_list)
    cal_res_list_df.to_excel(os.path.join(main_dir,'cal_res_list_df.xlsx') ,index=False)
    