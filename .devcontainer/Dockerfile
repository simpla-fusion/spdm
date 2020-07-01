
# FROM fy_imas_actors:latest as imas_stage

# FROM fylab:latest as fylab_stage

# FROM alpine:latest as merge_stage

# COPY --from=fylab_stage /fuyun/modules         /fuyun/modules
# COPY --from=fylab_stage /fuyun/software        /fuyun/software
# COPY --from=fylab_stage /fuyun/ebfiles_repo    /fuyun/ebfiles_repo

# COPY --from=imas_stage  /fuyun/modules         /fuyun/modules
# COPY --from=imas_stage  /fuyun/software        /fuyun/software
# COPY --from=imas_stage  /fuyun/ebfiles_repo    /fuyun/ebfiles_repo

# FROM fybase:latest

# ENV FYDEV_USER fydev

# COPY --chown=${FYDEV_USER}:${FYDEV_USER} --from=merge_stage /fuyun/modules         /fuyun/modules
# COPY --chown=${FYDEV_USER}:${FYDEV_USER} --from=merge_stage /fuyun/software        /fuyun/software
# COPY --chown=${FYDEV_USER}:${FYDEV_USER} --from=merge_stage /fuyun/ebfiles_repo    /fuyun/ebfiles_repo
FROM registry.cn-hangzhou.aliyuncs.com/fusionyun/fylab:v0.0.1-imas

USER ${FYDEV_USER}

ENV MODULEPATH=${FUYUN_DIR}/modules/base:${FUYUN_DIR}/modules/compiler:${FUYUN_DIR}/modules/data:${FUYUN_DIR}/modules/devel:${FUYUN_DIR}/modules/lang:${FUYUN_DIR}/modules/lib:${FUYUN_DIR}/modules/math:${FUYUN_DIR}/modules/mpi:${FUYUN_DIR}/modules/numlib:${FUYUN_DIR}/modules/system:${FUYUN_DIR}/modules/toolchain:${FUYUN_DIR}/modules/tools:${FUYUN_DIR}/modules/vis:${FUYUN_DIR}/modules/phys:${FUYUN_DIR}/modules/fuyun

RUN mkdir -p /home/${FYDEV_USER}/.vscode-server/extensions

RUN source /etc/profile.d/modules.sh && \
    module load Python && \
    pip install --upgrade pip && \
    pip install pylint autopep8
