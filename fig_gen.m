load('for_matlab.mat')
close all;

ana_2R_kHz = formula2Rs / 1e3;
ana_2E_kHz = formula2Es / 1e3;
got_2R_kHz = 1e6./t2rs;
got_2E_kHz = 1e6./t2es;

% x = log10(x_Phi0s/1e-6);
x = x_Phi0s/1e-6;

% figure(1)
y1= (ana_2E_kHz);
y2= (got_2E_kHz);
% plot(x,y1,'--')
% plot(x,y2,'o')
% loglog(x,y1,'--',x,y2,'o')

figure1 = figure(1);
z1= (ana_2R_kHz);
z2= (got_2R_kHz);
loglog(x,y1,'--b',x,y2,'ob',x,z1,'--r',x,z2,'or')
legend( ...
    'Echo 1st order analytical','Echo numerical', ...
    'Ramsey 1st order analytical','Ramsey numerical', ...
    Location='southeast')
xlabel('\Phi - \Phi_0/2 (\mu\Phi_0)')
ylabel('dephasing rate (kHz)')

annotation(figure1,'textbox',...
    [0.157671957671958 0.696428571428571 0.228902116402116 0.204059523809524],...
    'String',{'\Delta = 5 GHz','I_p = 300 nA',sprintf('A_\\Phi = %.1f \\mu\\Phi_0',A_Phi0*1e6)},...
    'LineWidth',2,...
    'FontSize',20.5,...
    'FitBoxToText','off');