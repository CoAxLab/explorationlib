
# Parameter sweeps in an attempt to reproduce the results of 
# 1.Viswanathan, G. M. et al. Optimizing the success of random searches. 
# Nature 401, 911–914 (1999).

# Levy2 v Diffusion 1

# 5000 targets
viswanathan5000a:
	python ../explorationlib/run.py experiment viswanathan5000a_levy2.pkl Levy2d ViswanathanField5000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan5000a_diff.pkl Diffusion2d ViswanathanField5000 --num_steps=1000 --num_experiments=1000

viswanathan5000b:
	python ../explorationlib/run.py experiment viswanathan5000b_levy2.pkl Levy2d ViswanathanField5000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan5000b_diff.pkl Diffusion2d ViswanathanField5000 --num_steps=100 --num_experiments=1000

viswanathan5000c:
	python ../explorationlib/run.py experiment viswanathan5000c_levy2.pkl Levy2d ViswanathanField5000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan5000c_diff.pkl Diffusion2d ViswanathanField5000 --num_steps=100 --num_experiments=1000

# 50000 targets
viswanathan50000a:
	python ../explorationlib/run.py experiment viswanathan50000a_levy2.pkl Levy2d ViswanathanField50000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan50000a_diff.pkl Diffusion2d ViswanathanField50000 --num_steps=100 --num_experiments=1000

viswanathan50000b:
	python ../explorationlib/run.py experiment viswanathan50000b_levy2.pkl Levy2d ViswanathanField50000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan50000b_diff.pkl Diffusion2d ViswanathanField50000 --num_steps=100 --num_experiments=1000

viswanathan50000c:
	python ../explorationlib/run.py experiment viswanathan50000c_levy2.pkl Levy2d ViswanathanField50000 --num_steps=100 --num_experiments=1000
	python ../explorationlib/run.py experiment viswanathan50000c_diff.pkl Diffusion2d ViswanathanField50000 --num_steps=100 --num_experiments=1000

