config=$1

module load python/3.8

module load scipy-stack

# create venv and install requirements.
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt 

python3 -u run_iccea_mtq_w_args.py configs.$config