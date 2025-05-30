function [YELMtrain,Perform,TrainedELM] = ELMRegularizedtrainRainfall(Ycal,Xcal,NoofHidNeu,ActivationFunction,Options,RegCoeff)
RMSEC=nan(length(RegCoeff),1);
CorrC=nan(length(RegCoeff),1);
if length(RegCoeff)>1
    for i=1:length(RegCoeff)
        [~,Perform,~] = ELMRegularizedtrainSingleCoeff(Ycal,Xcal,NoofHidNeu,ActivationFunction,Options,RegCoeff(i));
        RMSEC(i)=Perform.RMSE;
        CorrC(i)=Perform.Corr;
    end
    TF=RMSEC==min(RMSEC) & CorrC>0;
    if sum(TF)==0
            TF=RMSEC==min(RMSEC);
    end
    OptRegCoeff=RegCoeff(TF);
    OptRegCoeff=OptRegCoeff(1);
    [YELMtrain,Perform,TrainedELM] = ELMRegularizedtrainSingleCoeff(Ycal,Xcal,NoofHidNeu,ActivationFunction,Options,OptRegCoeff);
    TrainedELM.OptRegCoeff=OptRegCoeff;
else
    [YELMtrain,Perform,TrainedELM] = ELMRegularizedtrainSingleCoeff(Ycal,Xcal,NoofHidNeu,ActivationFunction,Options,RegCoeff);
    TrainedELM.OptRegCoeff=RegCoeff;
end
end

function [YELMtrain,Perform,TrainedELM] = ELMRegularizedtrainSingleCoeff(Ycal,Xcal,NoofHidNeu,ActivationFunction,Options,RegCoeff)
%     ActivationFunction='sig';
Perform.RMSE=nan;
Perform.Corr=nan;
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
%   OutputWeight=pinv(H') * Ycal';
%     OutputWeight=(H') \ Ycal';
C = RegCoeff;
% OutputWeight=inv( eye(size(H,1))/C + H * H') * H * Ycal';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
firstTerm =eye(size(H,1))/C + H * H';
OutputWeight= (firstTerm \ H) * Ycal';   % faster method 1 //refer to 2012 IEEE TSMC-B paper

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
