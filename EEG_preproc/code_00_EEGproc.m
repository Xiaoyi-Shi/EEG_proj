%%% EEG数据预处理脚本
%This is a manually executed script for preprocessing EEG data.
%input: data_00_EEGraw (edf format)
%output: data_00_EEGraw/_badseg (csv format)
%%%

raw_file = 'sub074.edf';
dest_file = '';

cfg = [];
cfg.dataset       = raw_file;
cfg.lpfilter       = 'yes';
cfg.lpfreq         = 50;
cfg.hpfilter       = 'yes';
cfg.hpfreq         = 1;
cfg.bsfilter       = 'yes';
cfg.bsfiltord      = 3;
cfg.bsfreq         = [49 51; 99 101; 149 151; 199 201];
cfg.channel       = 1:19;
data_raw          = ft_preprocessing(cfg);
%分段(可选)
cfg = [];
cfg.length = 5; % 每段的长度为 5 秒
cfg.overlap       = 0; % 50 percent overlap
data_segmented = ft_redefinetrial(cfg, data_raw);

% 去坏导坏段
cfg          = [];
cfg.method   = 'summary';
cfg.keepchannel = 'yes';
cfg.keeptrial   = 'yes';
%cfg.viewmode    = 'remove';
%cfg.ylim     = [-1e-12 1e-12];
data_dummy = ft_rejectvisual(cfg, data_segmented);

%导出坏段信息
bad_segment = data_dummy.cfg.artfctdef.summary.artifact;
writematrix(bad_segment, replace(raw_file, '.edf', '_badseg.csv'));


%浏览原始波形
cfg          = [];
cfg.viewmode = 'vertical';
cfg.ylim     = [-40 40];
ft_databrowser(cfg, data_raw);

%浏览去坏段后波形
cfg          = [];
cfg.viewmode = 'vertical';
cfg.ylim     = [-40 40];
ft_databrowser(cfg, data_dummy);