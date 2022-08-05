#!/bin/bash
#SBATCH --job-name=arc
#SBATCH --account=ec12
#SBATCH --mail-type=FAIL
#SBATCH --time=05:00:00
#SBATCH --partition=accel --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --ntasks-per-node=2


#
# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
#
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module load Python/3.7.4-GCCcore-8.3.0
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-python-candy/2021.01-foss-2019b-Python-3.7.4
module load nlpl-pytorch/1.7.1-foss-2019b-cuda-11.1.1-Python-3.7.4
module load nlpl-datasets/1.17-foss-2019b-Python-3.7.4:
module load nlpl-transformers/4.5.1-foss-2019b-Python-3.7.4

file=$1
# print information (optional)
echo "submission directory: ${SUBMITDIR}"
ulimit -a
module list

# by default, pass on any remaining command-line options
echo $file
chmod u+x $file
./$file
