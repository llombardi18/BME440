% EEG information regarding movement vs no movement
addpath(genpath("BCI2000_FILES"), '-frozen')
addpath(genpath("lib"))

clearvars
    
runs = [4,8,12];

Dataset = [];

nparticipants = 50;
for IDparticipant=1:nparticipants
    
    taskLeftHand_run = zeros(64,3);
    taskRightHand_run = zeros(64,3);

    subjectString = ['S' , num2str(IDparticipant, '%03d')];
    fullpath = ['.\BCI2000_FILES\', subjectString];
    files = dir([fullpath '\*.dat"']);
    
    for IDrun = runs %these should be odd numbers for hand
        
        filenameWithPath = [fullpath, '\', files(IDrun).name];

        [signal, states, parameters] = load_bcidat(filenameWithPath,'-calibrated'); 
        nTimePoints = size(signal,1);
        nElectrodes = size(signal,2);
        
        fs = str2double(parameters.SamplingRate.Value{1});
        tContinuous = [0: 1/fs: (nTimePoints-1)/fs];
        
        hp = designfilt('highpassiir','StopbandFrequency', 10, 'PassbandFrequency',12, ...
            'PassbandRipple',1,'SampleRate',fs);
        
        lp = designfilt('lowpassiir','StopbandFrequency', 30, 'PassbandFrequency', 25, ...
            'PassbandRipple',1,'SampleRate',fs);
        
        signal = filtfilt(lp,signal);
        signal = filtfilt(hp,signal);
        
        % Spatial filtering: common average referencing.
        CAR = mean(signal,2); %average over electrode
        signal = signal - CAR;
        
        %power
        power = envelope(signal);

%         figure;
%         plot(signal(1:1000,1))
%         hold on
%         plot(power(1:1000,1))

        %separate the trials
        restLocations = (round(states.TargetCode) == 0); %
        activeLocations_RightHand = (round(states.TargetCode) == 2);
        activeLocations_LeftHand = (round(states.TargetCode) == 1);

        baseline = mean(power(restLocations,:),1);
        taskLeftHand = mean(power(activeLocations_LeftHand,:),1);
        taskRightHand = mean(power(activeLocations_RightHand,:),1);
        
        taskLeftHand_normalized = taskLeftHand - baseline;
        taskRightHand_normalized = taskRightHand - baseline;
        

%         figure
%         subplot(1,3,1)
%         topoplot(baseline,'eloc64.txt','eeg',...
%                         'colormap',redblue(100),'maplimits', [0 10])
%         subplot(1,3,3)
%         topoplot(taskLeftHand,'eloc64.txt','eeg',...
%                         'colormap',redblue(100),'maplimits', [0 10])
% 
%         subplot(1,3,2)
%         topoplot(taskRightHand,'eloc64.txt','eeg',...
%                         'colormap',redblue(100),'maplimits', [0 10])
%         
%         
%         figure
%         subplot(1,2,1)
%         topoplot(taskLeftHand_normalized,'eloc64.txt','eeg',...
%                         'colormap',redblue(100),'maplimits', [-10 10])
%         subplot(1,2,2)
%         topoplot(taskRightHand_normalized,'eloc64.txt','eeg',...
%                         'colormap',redblue(100),'maplimits', [-10 10])

        taskLeftHand_run(:,IDrun) = taskLeftHand_normalized;
        taskRightHand_run(:,IDrun) = taskRightHand_normalized;

    end%runs

    %average over runs
    X_left = mean(taskLeftHand_run,2);
    X_right = mean(taskRightHand_run,2);

%     figure
%     subplot(1,2,1)
%     topoplot(X_left,'eloc64.txt','eeg',...
%                     'colormap',redblue(100),'maplimits', [-1 1])
%     subplot(1,2,2)
%     topoplot(X_right,'eloc64.txt','eeg',...
%                     'colormap',redblue(100),'maplimits', [-1 1])
    
    
    Dataset = cat(1,Dataset,[X_left' 0]);
    Dataset = cat(1,Dataset,[X_right' 1]);

end

writematrix(Dataset,'Data/DatasetEEGProject.csv');
