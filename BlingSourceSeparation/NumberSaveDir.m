function SaviDirNumbered = NumberSaveDir(SaviDir)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Assign the sequential number to the directory where files are saved.
%
% Input
% SaveDir:	Directory where the results are saved
% 
% Output
% SaveDir:	Directory with number
%__________________________________________________________________________
n = 1;
while true
	SaviDirNumbered = [SaviDir,'_',num2str(n)];
	FlagMake = mkdirIfNotExist(SaviDirNumbered);
	if FlagMake; break;
	else; n=n+1; end
end
end