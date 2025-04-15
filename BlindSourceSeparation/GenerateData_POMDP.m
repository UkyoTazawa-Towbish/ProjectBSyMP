function P = GenerateData_POMDP(A,B,D,T,FlagFig,ShowProgress)
%__________________________________________________________________________
% This code was created for the following work.
% Synaptic pruning facilitates online Bayesian model selection
% Ukyo T. Tazawa, Takuya Isomura
%
% Copyright (C) 2024 Ukyo Towbish Tazawa
%
% 2024-06-06
%__________________________________________________________________________
% Subroutine of GenerateTask_BSS.m. Generate data through POMDP.
%
% Input
% A:			A-matrix (likelihood mapping matrix) in the POMDP model
% B:			B-matrix (state transitoin matrix) in the POMDP model
% D:			D-vector (prior for the initial states) in the POMDP model
% T:			Total time steps of a session
% FlagFig:		Flag to switch whether to make figures or not
% ShowProgress:	Flag to switch whether to show progress or not
% 
% Output
% P:			Info of outer world (truth)
%__________________________________________________________________________
if ShowProgress
	f = waitbar(0,'Generating data...');
	f.NumberTitle = 'off';
	f.Name = 'Progress Status';
end
P = struct; P.A = A; P.B = B; P.D = D; P.T = T;
P.No = size(A,3); P.Ns = size(A,4);
P.s = NaN(2,P.Ns,P.T);
P.o = NaN(2,P.No,P.T);

t = 1;
P.s(1,:,t) = P.D(1,:)>rand(size(P.D(1,:))); % P(s_1|D)=Cat(D)
P.s(2,:,t) = 1-P.s(1,:,t);
P.sProb(:,:,t) = P.D;
P.o(:,:,t) = Categ(P.A,P.s(:,:,1)); % P(o_t|s_t,A)=Cat(As_t)

for t=2:P.T
	if ShowProgress; waitbar(t/P.T,f); end
	[P.s(:,:,t),P.sProb(:,:,t)] = Categ(P.B,P.s(:,:,t-1));	% P(s_t|s_t-1,B)=Cat(Bs_t-1)
	P.o(:,:,t) = Categ(P.A,P.s(:,:,t));						% P(o_t|s_t,A)=Cat(As_t)
end
if ShowProgress; close(f); end

if FlagFig
	F = struct;
	F.s = squeeze(P.s(1,:,:)); F.o = squeeze(P.o(1,:,:));
	F.A_causality = squeeze(var(P.A./sum(P.A,1),0,[1,2]));
	F.B_causality = squeeze(var(P.B./sum(P.B,1),0,[1,2]));
	
	figure();
	h = heatmap(F.s);
	xlabel('time'); ylabel('state');
	nLabels = 10;
	XLabelsNum = 1:P.T;
	XLabels = string(XLabelsNum);
	XLabels(mod(XLabelsNum,nLabels) ~= 0) = " ";
	h.XDisplayLabels = XLabels;
	title('Timecourse of hidden state');
	
	figure();
	h = heatmap(F.o);
	xlabel('time'); ylabel('observation');
	h.XDisplayLabels = XLabels;
	title('Timecourse of observation');
	
	figure();
	heatmap(F.A_causality);
	title('Causality of each element in A');
	
	figure();
	heatmap(F.B_causality);
	title('Causality of each element in B');
end
end


%% nested functions
function [VecOut,MatProb] = Categ(Mat,VecIn) % Categorical distribution
a = pagemtimes(Mat,repmat(reshape(VecIn,2,1,1,[]),[1,1,size(Mat,3),1]));
MatProb = squeeze(prod(a,4))./sum(squeeze(prod(a,4)),1);
VecOut = NaN(size(MatProb));
VecOut(1,:) = MatProb(1,:)>rand(size(MatProb(1,:)));
VecOut(2,:) = 1-VecOut(1,:);
end