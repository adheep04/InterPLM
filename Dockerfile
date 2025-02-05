# Use Miniconda as the base image
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git

# Set working directory
WORKDIR /app

# Clone the InterPLM repository
RUN git clone https://github.com/ElanaPearl/interPLM.git .

# Create the conda environment from env.yml
RUN conda env create -f env.yml

# Update PATH so that the 'interplm' environment’s binaries come first.
ENV PATH /opt/conda/envs/interplm/bin:$PATH

# Install the InterPLM package in editable mode
RUN pip install -e .

# Expose port 8501 for the dashboard
EXPOSE 8501

# Set the default command to launch bash
CMD ["bash"]
