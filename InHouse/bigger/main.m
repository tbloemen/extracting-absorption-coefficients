clear all
close all

%%%%%%%%
% Room %
%%%%%%%%
L = [8.1, 6.8, 3.07]; %[m], [Lx, Ly, Lz]

%The loudspeaker locations. The files are numbered per loudspeaker. 
%Note that the numbers correspond to the channel of the fireface, which is also why some numbers are missing.
%FOr the open curtains, I only have data of loudspeaker 3
loc_loud = [     3.5600    3.9050    1.1000;    %1  
                5.5600    2.0190    1.1750;     %3
                0.3150    2.7450    1.1750;     %4
                2.9200    1.6300    1.0800;     %5
                5.4760    5.3300    1.0800;     %6
                1.9320    5.3300    1.2750];    %7

true_loc_loud = [4.0100    3.9050    1.1000;
    6.0100    2.0190    1.1750;
    0.7650    2.7450    1.1750;
    3.3700    1.6300    1.0800
    5.9260    5.3300    1.0800
    2.3820    5.3300    1.2750]


%The microphone locations. The numbering correspond to the number of channels of the RIR. Each RIR is a real matrix of size N x 8, 
%   such that the first row contains the RIR from the loudspeaker to microphone 1, the second to microphone 2, etc.
loc_mic = [ 1.6600    3.3300    0.9900;     %1
            1.6307    3.2593    0.9900;     %2
            1.5600    3.2300    0.9900;     %3
            1.4893    3.2593    0.9900;     %4
            1.4600    3.3300    0.9900;     %5
            1.4893    3.4007    0.9900;     %6
            1.5600    3.4300    0.9900;     %7
            1.6307    3.4007    0.9900];    %8

