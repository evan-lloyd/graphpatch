FROM graphpatch-dev
WORKDIR /graphpatch

# Pre-create test environments for, not the full tox matrix, but one per torch version, to bootstrap
# the most common use case for this image (validating torch-version-dependent behavior).
RUN tox -m test-20 --notest
RUN tox -m test-21 --notest
RUN tox -m test-22 --notest
RUN tox -m test-23 --notest
RUN tox -m test-24 --notest
RUN tox -m test-25 --notest
