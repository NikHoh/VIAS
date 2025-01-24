FROM sapmachine:jdk-ubuntu-24.04

ARG OSMOSIS_URL="https://github.com/openstreetmap/osmosis/releases/download/0.49.2/osmosis-0.49.2.tar"
ENV OSMOSIS_URL=$OSMOSIS_URL

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl python3 python3-pip git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install osmosis (needed for met.py)
RUN set -x \
  && useradd -ms /bin/bash vias_user \
  && mkdir -p /opt/osmosis \
  && curl -L $OSMOSIS_URL | tar -vxf - -C /opt/osmosis --strip-components=1 \
  && ls -l /opt/osmosis/bin \
  && ln -s /opt/osmosis/bin/osmosis /usr/local/bin/osmosis \
  && chmod a+x /usr/local/bin/osmosis


# Create the data directory in the container
RUN set mkdir -p /VIAS_data

# Set the working directory inside the container
WORKDIR /VIAS_pub
COPY . /VIAS_pub

# Install VIAS package
RUN pip3 install --disable-pip-version-check --break-system-packages -e .

# Set example working directory
WORKDIR /VIAS_pub/examples

USER vias_user

# Default CMD to display help if no script is provided
CMD ["python3", "-c", "print('Specify a script to run with python3: met_example.py, mct_example.py, mopp_example.py, or pvt_example.py')"]