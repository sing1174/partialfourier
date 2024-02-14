This is an implementation of GRAPPA.

GRAPPA_1.m is implemented on 120x128x5 dimension data, 5 is number of channels.
Our data is full K-space data. We create a mask for R=2 (sampling rate) with acs lenght = 12.
Then run GRAPPA to fill the missing k-space data.
Next, we mask the original data and run GRAPPA on reconstructed lines to see the output. 

GRAPPA_2.m is different implemenatation of GRAPPA on 96x96x32 data. 
