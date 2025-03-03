function MatFiles2CSV(DirParent,CellMatFiles,SaveDir)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Load mat-files and write the content to csv-files.
%
% Input
% DirParent:	Parent directory where the mat-files are located
% CellMatFiles:	Cell containing file-names of mat-files
% SaveDir:		Directory where the csv-files will be saved
%__________________________________________________________________________
if nargin < 3 % if SaveDir is not specified
	SaveDir = DirParent;
end
for i = 1:length(CellMatFiles)
	DirCurFile = fullfile(DirParent,CellMatFiles{i});
	load(DirCurFile);
	CellVarNamesCur = who('-file',DirCurFile);
	for j = 1:length(CellVarNamesCur)
		eval(['mat2csv(SaveDir,',CellVarNamesCur{j},')']);
	end
end
end

function mat2csv(SaveDir,var)
name = inputname(2);
writematrix(var,fullfile(SaveDir,[name,'.csv']));
end