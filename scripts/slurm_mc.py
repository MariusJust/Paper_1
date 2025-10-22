import os

ram_allocated = 200                      # change allocated ram

def job_func(job_name):
    s = '#!/bin/bash\n'
    s += '#SBATCH --job-name={}\n'.format(job_name)
    s += '#SBATCH --partition=q36\n'    # check gnodes
    s += '#SBATCH --mem=200G\n'          # change allocated ram
    s += '#SBATCH --nodes=1\n'
    s += '#SBATCH --time=90:00:00\n'   # change estimated time
    s += '#SBATCH --ntasks-per-node=1\n'
    s += '#SBATCH --cpus-per-task=36\n'
    s += 'cd $SLURM_SUBMIT_DIR\n'
    s += 'export PYTHONPATH=$SLURM_SUBMIT_DIR/src:$PYTHONPATH\n'
     # ─── Export TF/CUDA vars here ─────────────────────────────────────
    s += 'export TF_ENABLE_ONEDNN_OPTS=0\n'
    s += 'export TF_CPP_MIN_LOG_LEVEL=3\n'
    s += 'export CUDA_VISIBLE_DEVICES=""\n'
    s += 'python -u src/simulations/monte_carlo.py 2>&1   | grep -Ev "All log messages before absl::InitializeLog|Unable to register cuDNN factory|Unable to register cuBLAS factory"\n'
    return s


name = f'MC' # name of the job in Grendel
f = open('slurm.job', 'w')
f.write(job_func(name))
f.close()
#os.system('sbatch slurm.job')
os.system('sbatch --mem={}G slurm.job'.format(ram_allocated))
    
    

