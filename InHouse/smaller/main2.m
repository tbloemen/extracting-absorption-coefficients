%Note:  (a) the RIRs are normalised such that the highest peak equals one
%       (b) there is not corrected for stuff such as loudspeaker and microphone transfer. 
%           Maybe it is interesting to note that the loudspeaker is single-cone, so I dont think it is that much of an issue 
%       (c) The time till the first reflection (direct path) arrives might not be accurate:
%           there might be delauys in amplifier, analog-to-digital conversion, etc.

clear all
close all

[RIRopen, Fs] = audioread('curtains_open_real_room_3.wav');
[RIRclosed, Fs] = audioread('curtains_closed_real_room_3.wav');

L = [8.1, 6.8, 3.07];           %[m], Lx, Ly, Lz 
LS_loc = [3.37, 1.63, 1.08];    %[m], Loudspeaker location
MIC_loc = [1.66, 3.33, 0.99];   %[m], Microphone location

%Some plotting stuff
t_ax_closed = 0:1/Fs:length(RIRclosed)/Fs-1/Fs;
t_ax_open = 0:1/Fs:length(RIRopen)/Fs-1/Fs;
figure
plot(t_ax_open, RIRopen)
hold on
plot(t_ax_closed, RIRclosed)
xlabel('Time [s]')
ylabel('Amplitude [-]')
legend('Curtains open (T60 ~ 0.42 s)', 'Curtains closed (T60 ~ 0.22 s)');
xlim([0, 0.5])


