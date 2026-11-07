FROM python@sha256:c845af9399020c7e562969a13689e929074a10fd057acd1b1fad06a2fb068e97
COPY duckdb-2.0.0-cp312-abi3-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl pytz-2026.6-py2.py3-none-any.whl /wheel/
RUN echo 'e38daf0acbad9de7ce892d402b68409519291ee16af3c2cba85ace3b085c7eaa  /wheel/duckdb-2.0.0-cp312-abi3-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl' | sha256sum -c - && echo '7fed3fee8fd6bd4a2b7b75a8d56021d0624127bf69bd5691f115db929ac570b3  /wheel/pytz-2026.6-py2.py3-none-any.whl' | sha256sum -c - && python -I -B -m pip install --no-index --no-deps --no-cache-dir /wheel/*.whl && rm -r /wheel
ARG WORKER_BUNDLE
LABEL org.strata.sql.profile="sql_local_arm64_v1" org.strata.sql.bundle="${WORKER_BUNDLE}"
COPY _sql_worker.py _sql_wire.py /app/
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "/app/_sql_worker.py"]
