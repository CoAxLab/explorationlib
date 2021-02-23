
# Parameter sweeps in an attempt to reproduce the results of 
# 1.Viswanathan, G. M. et al. Optimizing the success of random searches. 
# Nature 401, 911–914 (1999).

# Levy2 v Diffusion 1

# 5000 targets
viswanathan5000:
	python ../explorationlib/run.py experiment viswanathan5000_levy2.pkl Levy2d ViswanathanField5000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan5000_diff.pkl Diffusion2d ViswanathanField5000 --num_steps=1000 --num_experiments=1000

# 50000 targets
viswanathan50000:
	python ../explorationlib/run.py experiment viswanathan50000_levy2.pkl Levy2d ViswanathanField50000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan50000_diff.pkl Diffusion2d ViswanathanField50000 --num_steps=100 --num_experiments=1000
