# Sandbox for TUFF (web and files profiles). Built and run with Apple `container`: by
# Scripts/sandbox_selftest.sh for development, and by the GitHub Actions
# workflow .github/workflows/sandbox-image.yml for the published image.
#
# Everything the image holds is pinned, so a rebuild installs the same code:
# - the base image by digest (python:3.12-slim);
# - Debian packages by a snapshot date on snapshot.debian.org (DEBIAN_SNAPSHOT),
#   so the nftables that loads the firewall is the same on every build;
# - every Python package by hash (requirements.txt, generated from
#   requirements.in as that file describes).
# To update, change the digest, the snapshot date and requirements.txt together.
# The build prints the installed versions of nftables and Python so a build log
# records them. Check the Python patch level there against CVE-2025-6069
# (html.parser, quadratic) when the digest is updated; the server also runs its
# HTML fallback parser in a killed-after-timeout child.
FROM docker.io/library/python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016

ARG DEBIAN_SNAPSHOT=20261001T000000Z

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /app/requirements.txt
# The RUN below points apt at the snapshot (deb822 and one-line formats) and
# refuses to go on if any source still names the live archive.
RUN set -eux; \
    for file in /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list; do \
      [ -f "$file" ] || continue; \
      sed -i \
        -e "s#https\?://deb.debian.org/debian-security#http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}#g" \
        -e "s#https\?://deb.debian.org/debian#http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}#g" \
        -e "s#https\?://security.debian.org/debian-security#http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}#g" \
        "$file"; \
    done; \
    if grep -rE 'deb\.debian\.org|security\.debian\.org' /etc/apt; then \
      echo "an apt source still names the live Debian archive" >&2; exit 1; \
    fi; \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99snapshot; \
    apt-get update; \
    apt-get install -y --no-install-recommends nftables; \
    dpkg-query --show nftables; python --version; \
    rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99snapshot; \
    pip install --require-hashes --no-deps -r /app/requirements.txt; \
    useradd --uid 10001 --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin sandbox

COPY server.py firewall-web.nft firewall-files.nft entrypoint.sh default_gateway.py /app/

# One image for both profiles (SANDBOX_PROFILE=web or files, see server.py).
# Starts as root only to load the profile's firewall and read the secret; server.py
# then switches to user 10001 and drops every capability before it serves a
# request. The app mounts the secret read-only at /run/tuff-sandbox/secret.
EXPOSE 9100
ENTRYPOINT ["/app/entrypoint.sh"]
