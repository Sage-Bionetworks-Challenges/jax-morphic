# Define the base of your image.
# We recommend to always specify a tag. `latest` may not 
# always ensure reproducibility.
FROM python:3.11.8

# For best practice, run container as non-root user.
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

# Set the working directory for the COPY, RUN, and ENTRYPOINT
# commands of the Dockerfile.
WORKDIR /home/user

# Copy files over to the image.
# We recommend copying over each file individually, as to take
# advantage of cache building (which helps reduce build time).
COPY --chown=user:user requirements.txt .

# Install needed libraries/packages.
# Your model will be run without network access, so the dependencies
# must be installed here (and not during code execution).
RUN pip install --upgrade pip && \
    pip install \
    --user \
    --no-cache-dir \
    --break-system-packages \
    -r requirements.txt

COPY --chown=user:user run_model.py .

# Set the main command of the image.
# ENTRYPOINT is the executable; CMD supplies default arguments that can be
# overridden at `docker run` time, e.g.:
#   docker run <image> --nthreads 16 --n-feats 5000
ENTRYPOINT ["python", "run_model.py"]
CMD ["--nthreads", "8", "--n-feats", "10000"]