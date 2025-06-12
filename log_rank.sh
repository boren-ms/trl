#! /bin/bash
# dist_train.sh
set -euo pipefail

RANK=${MPI_RANK:-0}
RANK_LOG_FILE=${RCALL_LOGDIR}/*/rank_${RANK}.log
echo "[${RANK} ] Tail ${RANK_LOG_FILE}"
tail -n 30 $RANK_LOG_FILE