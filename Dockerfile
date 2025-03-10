# Use a Debian-based Miniconda image
FROM continuumio/miniconda3

# Install Java (OpenJDK 17) and clean up
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy Conda environment file and create environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Use Conda environment for shell
SHELL ["conda", "run", "-n", "myapp_env", "/bin/bash", "-c"]

# Copy all app files
COPY . .

# Make sure scripts are executable
RUN chmod +x scripts/run.sh
RUN chmod +x scripts/entrypoint.sh

# Set flexible entrypoint
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
