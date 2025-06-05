import os

ram_allocated = 200                      # change allocated ram

def job_func(job_name):
    s = '#!/bin/bash\n'
    s += '#SBATCH --job-name={}\n'.format(job_name)
    s += '#SBATCH --partition=q64\n'    # check gnodes
    s += '#SBATCH --mem=200G\n'          # change allocated ram
    s += '#SBATCH --nodes=1\n'
    s += '#SBATCH --time=90:00:00\n'   # change estimated time
    s += '#SBATCH --ntasks-per-node=1\n'
    s += '#SBATCH --cpus-per-task=55\n'
    s += 'cd $SLURM_SUBMIT_DIR\n'
    s += 'export PYTHONPATH=$SLURM_SUBMIT_DIR/src:$PYTHONPATH\n'
    s += 'python -u src/tuning/grid_search.py\n'
    return s


name = f'gs' # name of the job in Grendel
f = open('slurm.job', 'w')
f.write(job_func(name))
f.close()
#os.system('sbatch slurm.job')
os.system('sbatch --mem={}G slurm.job'.format(ram_allocated))
    
    
