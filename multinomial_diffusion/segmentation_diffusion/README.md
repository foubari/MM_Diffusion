First, make sure that everything is built (in the .../bct_generation/ directory):

``pip install .``


Then, to run the model:

``nohup python3.7 train.py --eval_every 1 --check_every 100 --kbest 3 --epochs 100 --name 'name_of_the_experiment' --device 'cuda' --diffusion_steps 1000 --dataset binary12 --batch_size 32 --dp_rate 0.1 --lr 0.0001 --warmup 5 ``

Here are some useful arguments:
``--diffusion-dim int`` : dimension of the features in the U-NET
``--eval_sample_every int``: to sample from the model during training
``--note_exp bool`` : whether to copy or not note_experiment.txt (that has to be in the same directory as train.py)

To resume from previous experiment:

``nohup python3.7 train.py --eval_every 1 --check_every 100 --epochs 200 --name 'name_of_the_experiment' --device 'cuda' --diffusion_steps 1000 --dataset binary12 --batch_size 32 --dp_rate 0.1 --lr 0.0001 --warmup 5 --resume True``


To sample from the model:

``python3.7 eval_sample.py --model '/export/fhome2/denis/bct_generation_data_and_results/multinomial_diffusion/log/\dataset\/multinomial_diffusion/multistep/\name_of_the_experiment\' --samples 10 --seed 0``

To inpaint from the model:

``python3.7 inpaint.py --model '/export/fhome2/denis/bct_generation_data_and_results/multinomial_diffusion/log/\dataset\/multinomial_diffusion/multistep/\name_of_the_experiment\' --originals 3 --samples 4 --seed 0``

NOTE:
For logging, it is for the moment assumed (without checks in the code) that check_every is divible by eval_every.