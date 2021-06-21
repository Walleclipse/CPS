# Modified stochastic gradient algorithms for controlling coherent pulse stacking

## Modified SG algorithms
The generic framework for stochastic gradient (SG) based algorithms.  
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="demo/generic_sgd.png"  width="400" ></div>  

Momentum (EMA of gradient) mechanism + SPGD = SPGD with momentum.       
Adaptive learning rate method + SPGD = SPGD with adaptive learning rate.    
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="demo/modified_spdg.png"  width="400" ></div>

## Result
The convergence speed of SPGD with momentum (and SPGD with adaptive learning rate) is faster than the original one. 
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="demo/comparison.png"  width="400" ></div>
