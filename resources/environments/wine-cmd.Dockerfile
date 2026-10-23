# 使用已有 MonoPlanner Python/Node Linux 镜像；仅增加 CMD 兼容运行环境。
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
RUN mkdir -p /tmp && chmod 1777 /tmp && apt-get update \
    && apt-get install -y --no-install-recommends wine=8.0~repack-4 wine64=8.0~repack-4 \
    && rm -rf /var/lib/apt/lists/*
ENV WINEARCH=win64 WINEPREFIX=/tmp/monoplanner-wine WINEDEBUG=-all WINEDLLOVERRIDES=mscoree,mshtml=
RUN wine --version
COPY compact-wine-prefix.py /opt/monoplanner/compact-wine-prefix.py
RUN WINEPREFIX=/opt/monoplanner/wine-template wineboot -u \
    && WINEPREFIX=/opt/monoplanner/wine-template wineserver -w \
    && python /opt/monoplanner/compact-wine-prefix.py
COPY wine /usr/local/bin/wine
RUN chmod 755 /usr/local/bin/wine
LABEL org.monoplanner.wine-cmd="1" \
      com.monoplanner.owner="wine-cmd-environment" \
      com.monoplanner.state=""
