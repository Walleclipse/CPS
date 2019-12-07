# Modified stochastic gradient algorithms for controlling coherent pulse stacking

Delay line coherent pulse stacking (CPS) is one of the promising methods to scale the peak power of the pulses, which use delay lines and phase modulators to stack up the pulses. Simple few-channel delay line CPS is straightforward to set up in the lab, but controlling the feedback of the delay line is more complex. Inspired by the success of exponential moving average (EMA) filter in the optimization of machine learning including momentum with SGD and adaptive learning rate with Adam, we introduce EMA techniques to CPS and create the algorithms called  SPGD with momentum and SPGD with adaptive learning rate. 
In this work, we open-source the delay line CPS simulation environment. Then we propose two modified SPGD algorithms that converge faster than original SPGD in the field of controlling feedback in CPS. 

