load('noise.mat')
sampleSize = 2000;
time = (1:sampleSize);
time = time * double(tuv)/1000;
sample = fq_freq_t(1:sampleSize)*1e6;
sample = sample - mean(sample);

figure(1)
plot(time,sample,'.-')
xlabel('time (\mus)')
ylabel('$f_{01}(t) - \left<f_{01}\right>$ (kHz)',Interpreter='latex')

figure(2)
histogram(fq_freq_t)