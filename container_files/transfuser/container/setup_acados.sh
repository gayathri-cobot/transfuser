#!/bin/bash
# acados RUNTIME environment only (no build-time codegen deps).
# cobot_local_nav_planner links libacados.so at runtime.

if [ -d /home/cobot/acados/lib ]; then
    export LD_LIBRARY_PATH="/home/cobot/acados/lib:${LD_LIBRARY_PATH:-}"
    export ACADOS_SOURCE_DIR=/home/cobot/acados
fi
