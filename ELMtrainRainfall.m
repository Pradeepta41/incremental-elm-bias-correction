function [YELMtrain,Perform,TrainedELM] = ELMtrainRainfall(Ycal,Xcal,NoofHidNeu,ActivationFunction,Options)
%     ActivationFunction='sig';

Perform.RMSE=nan;
Perform.Corr=nan;

if nargin == 4
    Options.Ymu=0;
    Options.Xmu=0;
    Options.Ystd=1;
    Options.Xstd=1;
end

Ymu=Options.Ymu;
Xmu=Options.Xmu;
Ystd=Options.Ystd;
Xstd=Options.Xstd;
Ycal=(Ycal-Ymu)/Ystd;     Xcal=(Xcal-Xmu)/Xstd;
Xcal=Xcal';
Ycal=Ycal';
NumberofTrainingData=size(Xcal,2);
NumberofInputNeurons=size(Xcal,1);
rng(1)
InputWeight=rand(NoofHidNeu,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NoofHidNeu,1)*2-1;      ind=ones(1,NumberofTrainingData);       BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
H=InputWeight*Xcal+BiasMatrix;
H=ActivationFunctionOutput(ActivationFunction,H);

%   OutputWeight = lsqminnorm(H',Ycal);
OutputWeight=pinv(H') * Ycal';
%     OutputWeight=(H') \ Ycal';
YELMtrain=(H' * OutputWeight)';
YELMtrain=YELMtrain*Ystd+Ymu;
Ycal=Ycal*Ystd+Ymu;

YELMmode=mode(YELMtrain);
absYELMtrain=abs(YELMtrain);
Offset=0;

% if min(absYELMtrain)~=min(Ycal)
%     if min(absYELMtrain)==YELMmode
%         Offset=min(Ycal) - min(absYELMtrain);
%     end
% end
YELMtrain=YELMtrain+Offset;
YELMtrain(YELMtrain<0)=0;     % Rainfall  MODIFICATION

%  Offset InputWeight  BiasMatrix  OutputWeight  Offset  ActivationFunction  NoofHidNeu
TrainedELM.InputWeight=InputWeight;
TrainedELM.BiasMatrix=BiasMatrix;
TrainedELM.OutputWeight=OutputWeight;
TrainedELM.Offset=Offset;
TrainedELM.ActivationFunction=ActivationFunction;
TrainedELM.NoofHidNeu=NoofHidNeu;

Perform.RMSE=sqrt(mse(Ycal,YELMtrain));
temp=corrcoef(Ycal,YELMtrain);
Perform.Corr=temp(1,2);
YELMtrain=YELMtrain';
end
% %%
% % figure; subplot(2,1,1); plot(rmseCal,'-+r'); hold on; plot(rmseVal,'--sb'); hold off; legend('Cal','Val'); ylabel('RMSE'); xlabel('Iterations');
%       subplot(2,1,2); plot(corCal,'-+r'); hold on; plot(corVal,'--sb'); ylabel('Correlation'); xlabel('Iterations');legend('Cal','Val')
