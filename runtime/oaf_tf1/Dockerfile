# check=skip=InvalidBaseImagePlatform
# The backend is intentionally frozen to the linux/amd64 base-image manifest.
FROM python@sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673 AS runtime-base

ARG RUNTIME_UID
ARG RUNTIME_GID

RUN set -eu; \
    case "${RUNTIME_UID}" in ""|0|65534|*[!0-9]*) exit 64;; esac; \
    case "${RUNTIME_GID}" in ""|0|65534|*[!0-9]*) exit 64;; esac; \
    groupadd --gid "${RUNTIME_GID}" crux; \
    useradd --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
      --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin crux

ENV HOME=/tmp \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TZ=UTC

FROM runtime-base AS runtime-build

COPY runtime/oaf_tf1/requirements.lock /opt/crux/requirements.lock
COPY runtime/oaf_tf1/wheelhouse/runtime/ /opt/crux/wheelhouse/
RUN python -m venv /opt/crux/venv; \
    /opt/crux/venv/bin/pip install --no-index --require-hashes --only-binary=:all: \
      --find-links=/opt/crux/wheelhouse \
      --requirement /opt/crux/requirements.lock

FROM runtime-build AS test

COPY runtime/oaf_tf1/requirements-test.lock /opt/crux/requirements-test.lock
COPY runtime/oaf_tf1/wheelhouse/test/ /opt/crux/test-wheelhouse/
RUN /opt/crux/venv/bin/pip install --no-index --require-hashes --only-binary=:all: \
      --find-links=/opt/crux/test-wheelhouse \
      --requirement /opt/crux/requirements-test.lock
COPY runtime/oaf_tf1/vendor/magenta/ /opt/crux/vendor/magenta/
COPY runtime/oaf_tf1/ /opt/crux/runtime/
WORKDIR /opt/crux
USER "${RUNTIME_UID}:${RUNTIME_GID}"

FROM runtime-base AS system-runtime-base

ARG DEBIAN_SNAPSHOT_URL
ARG DEBIAN_INRELEASE_SHA256
ARG DEBIAN_ARCHIVE_KEYRING_SHA256
ARG DEBIAN_SIGNING_FINGERPRINT
ARG DEBIAN_CODENAME
ARG DEBIAN_ARCHITECTURE
ARG SYSTEM_PACKAGE_MANIFEST_SHA256
ARG SYSTEM_PACKAGE_INVENTORY_SHA256

RUN set -eu; \
    case "${DEBIAN_SNAPSHOT_URL}" in ""|unset|none|sentinel) exit 64;; esac; \
    case "${DEBIAN_SIGNING_FINGERPRINT}" in \
      ""|*[!0-9A-F]*) exit 64;; \
    esac; \
    case "${#DEBIAN_SIGNING_FINGERPRINT}" in 40|64) :;; *) exit 64;; esac; \
    case "${DEBIAN_CODENAME}" in ""|*[!a-z0-9-]*) exit 64;; esac; \
    test "${DEBIAN_ARCHITECTURE}" = "amd64"; \
    for value in \
      "${DEBIAN_INRELEASE_SHA256}" \
      "${DEBIAN_ARCHIVE_KEYRING_SHA256}" \
      "${SYSTEM_PACKAGE_MANIFEST_SHA256}" \
      "${SYSTEM_PACKAGE_INVENTORY_SHA256}"; do \
        case "${value}" in ""|unset|none|sentinel|*[!0-9a-f]*) exit 64;; esac; \
        test "${#value}" -eq 64; \
    done
COPY tools/hpa320/oaf_system_packages.py /opt/crux/tools/oaf_system_packages.py
COPY runtime/oaf_tf1/system-packages/ /opt/crux/system-packages/
RUN python /opt/crux/tools/oaf_system_packages.py \
      --bundle /opt/crux/system-packages \
      --snapshot-url "${DEBIAN_SNAPSHOT_URL}" \
      --manifest-sha256 "${SYSTEM_PACKAGE_MANIFEST_SHA256}" \
      --inrelease-sha256 "${DEBIAN_INRELEASE_SHA256}" \
      --keyring-sha256 "${DEBIAN_ARCHIVE_KEYRING_SHA256}" \
      --inventory-sha256 "${SYSTEM_PACKAGE_INVENTORY_SHA256}" \
      --signing-fingerprint "${DEBIAN_SIGNING_FINGERPRINT}" \
      --codename "${DEBIAN_CODENAME}" \
      --architecture "${DEBIAN_ARCHITECTURE}" \
      --install; \
    rm -rf /opt/crux/system-packages /opt/crux/tools

FROM system-runtime-base AS runtime

COPY --from=runtime-build /opt/crux/venv /opt/crux/venv
COPY runtime/oaf_tf1/vendor/magenta/ /opt/crux/vendor/magenta/
COPY runtime/oaf_tf1/*.py /opt/crux/runtime/
COPY runtime/oaf_tf1/patches/ /opt/crux/runtime/patches/
COPY runtime/oaf_tf1/runner-source-manifest.json /opt/crux/runtime/runner-source-manifest.json
WORKDIR /opt/crux
USER "${RUNTIME_UID}:${RUNTIME_GID}"
ENTRYPOINT ["/opt/crux/venv/bin/python", "-I", "/opt/crux/runtime/entrypoint.py"]
