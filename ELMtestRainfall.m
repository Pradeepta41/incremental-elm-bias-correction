function [YELMtest,Perform] = ELMtestRainfall(Yval,Xval,TrainedELM,Options)
    Perform.RMSE=nan;
    Perform.Corr=nan;
    
    rng(1)
    InputWeight= TrainedELM.InputWeight;
    BiasMatrix=TrainedELM.BiasMatrix;
    OutputWeight=TrainedELM.OutputWeight;
    Offset=TrainedELM.Offset;
    ActivationFunction=TrainedELM.ActivationFunction;
%     NoofHidNeu=TrainedELM.NoofHidNeu;
%     Ymu=Options.Ymu;    Xmu=Options.Xmu;    Ystd=Options.Ystd;    Xstd=Options.Xstd;
%     Yval=(Yval-Ymu)/Ystd;     Xval=(Xval-Xmu)/Xstd;
    Xval=Xval'; 
    Yval=Yval'; 
    NumberofTestingData=size(Xval,2);
    ind=ones(1,NumberofTestingData);
    H=InputWeight.*Xval+BiasMatrix(:,ind);
    H=ActivationFunctionOutput(ActivationFunction,H);
    YELMtest=(H' .* OutputWeight)';
    YELMtest = YELMtest*Ystd+Ymu;     Yval=Yval*Ystd+Ymu;
    YELMtest=YELMtest + Offset;
    YELMtest(YELMtest<0)=0;    % Rainfall modification
    Perform.RMSE=sqrt(mean((Yval-YELMtest).^2));  %     Perform.RMSE=sqrt(mse(Yval,YELMtest));


    temp=corrcoef(Yval,YELMtest); 
    Perform.Corr=temp(1,2);
    YELMtest=YELMtest';
end
