function S = PreallocateStruct(ref,len)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Preallocate the space for structure array.
%
% Input
% ref:		Array to refer to (to be elongated)
% len:		Length of structure array.
%			"ref" will be elongated to the length of "len": length(S) = len
% 
% Output
% S:		Elongated structure array
%__________________________________________________________________________
if len == 1; S = ref; return; end
if isstruct(ref)
	S = ref;
    fields = fieldnames(ref);
elseif iscellstr(ref)
	S = struct();
    fields = ref;
else
	error('This function requires cell array containing character array.');
end
for i = 1:length(fields)
	S(len).(fields{i}) = [];
end
end