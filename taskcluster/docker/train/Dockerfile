FROM $DOCKER_IMAGE_PARENT
LABEL maintainer="Mozilla Release Engineering <release+docker@mozilla.com>"

RUN apt-get update -qq \
    && apt-get install -y python3-numpy \
                          python3-fasttext \
                          parallel \
                          zstd \
                          bc \
                          libhunspell-1.7-0 \
                          libboost-program-options1.74.0 \
    && apt-get clean

# Required to download sacrebleu datasets
RUN pip install sacrebleu

VOLUME /builds/worker/checkouts
