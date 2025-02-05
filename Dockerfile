###############################################################################
# Dockerfile for InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders
# This image sets up an environment for extracting, analyzing, and visualizing interpretable features from protein language models (PLMs) using sparse autoencoders (SAEs).

# Use Miniforge3 as the base image
FROM condaforge/miniforge3:latest

# Install system dependencies (wget and git) and clean up apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Clone the InterPLM repository into the container
RUN git clone https://github.com/ElanaPearl/interPLM.git .

# Create the conda environment from env.yml and clean conda caches
RUN conda env create -f env.yml && conda clean -a -y

# Update PATH so that the 'interplm' environment’s binaries come first.
ENV PATH /opt/conda/envs/interplm/bin:$PATH

# Install the InterPLM package in editable mode
RUN pip install -e .

# Expose port 8501 (for the dashboard)
EXPOSE 8501

# Set the default command to launch bash
CMD ["bash"]
