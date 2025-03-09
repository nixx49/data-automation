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
 
# Copy the Conda environment file
COPY environment.yml .
 
# Create the Conda environment
RUN conda env create -f environment.yml
 
# Use the Conda environment for all subsequent commands
SHELL ["conda", "run", "-n", "myapp_env", "/bin/bash", "-c"]
 
# Copy your application code into the container
COPY . .
 
# Set the entrypoint to run the application. The command here can be overridden.
ENTRYPOINT ["conda", "run", "-n", "myapp_env", "python", "src/run.py"]