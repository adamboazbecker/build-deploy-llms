# Step 1: Direct them to the right release
# Download them here: https://github.com/adamboazbecker/build-deploy-llms/releases

# Step 2: Build the image
docker build -t llms:1.0 .

# Step 3: Run the image with the files from the current directory mounted
docker run \
  -it \
  -v "$PWD:/app/." \
  llms:1.0 \
  /bin/bash
