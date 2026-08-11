# syntax=docker/dockerfile-upstream@sha256:3d6d54b33351b396a910d33248754b86b1d7dd838b4eeb9575d8903a209f6516
# check=skip=InvalidBaseImagePlatform
# The backend is intentionally frozen to the linux/amd64 base-image manifest.
FROM python@sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673 AS runtime-base

ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532

RUN set -eu; \
    case "${RUNTIME_UID}" in ""|0|65534|*[!0-9]*) exit 64;; esac; \
    case "${RUNTIME_GID}" in ""|0|65534|*[!0-9]*) exit 64;; esac; \
    groupadd --gid "${RUNTIME_GID}" crux; \
    useradd --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
      --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin crux

ENV CUDA_VISIBLE_DEVICES=-1 \
    PYTHONHASHSEED=0

FROM runtime-base AS runtime-build

COPY runtime/oaf_tf1/requirements.lock /opt/crux/requirements.lock
COPY runtime/oaf_tf1/wheelhouse/runtime/ /opt/crux/wheelhouse/
RUN python -m venv /opt/crux/venv; \
    /opt/crux/venv/bin/pip install --no-index --require-hashes --only-binary=:all: \
      --find-links=/opt/crux/wheelhouse \
      --requirement /opt/crux/requirements.lock

FROM runtime-build AS instrumented-source

COPY runtime/oaf_tf1/ /opt/crux/runtime/
COPY runtime/oaf_tf1/vendor/magenta/ /opt/crux/upstream/magenta/
COPY runtime/oaf_tf1/vendor/LICENSE /opt/crux/upstream/LICENSE
RUN python /opt/crux/runtime/apply_instrumentation_patch.py \
      --source /opt/crux/upstream \
      --destination /opt/crux/vendor \
      --patch /opt/crux/runtime/patches/capture-emitted-frame.patch \
      --manifest /opt/crux/runtime/instrumented-source-manifest.json

FROM runtime-build AS runtime

COPY --from=runtime-build /opt/crux/venv /opt/crux/venv
COPY --from=instrumented-source /opt/crux/vendor/ /opt/crux/runtime/vendor/
COPY --from=instrumented-source /opt/crux/upstream/ /opt/crux/upstream/
COPY runtime/oaf_tf1/*.py /opt/crux/runtime/
COPY runtime/oaf_tf1/model.json /opt/crux/runtime/model.json
COPY runtime/oaf_tf1/patches/ /opt/crux/runtime/patches/
WORKDIR /opt/crux
USER "${RUNTIME_UID}:${RUNTIME_GID}"
ENTRYPOINT ["/opt/crux/venv/bin/python", "-s", "/opt/crux/runtime/worker.py"]
