## Setup

Much of this is copied from the [can-scrapers prefect README](https://github.com/covid-projections/can-scrapers/blob/main/services/prefect/README.md).


### Make a new GCP compute instance
- SSH into it
- `sudo apt-get install --yes git git-lfs nginx`
- `git clone https://github.com/covid-projections/covid-data-model`
- `git clone https://github.com/covid-projections/covid-data-public`

Fix for 'ModuleNotFoundError: No module named '_ctypes'' during pip install: https://stackoverflow.com/a/62373223
- `sudo apt-get install libffi-dev`

Fix for 'Could not import the lzma module. Your installed Python is incomplete': https://stackoverflow.com/a/62407114
- `sudo apt-get install liblzma-dev lzma`

Install pyenv
- `sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils`
- `curl https://pyenv.run | bash`
- `cat >> ~/.bashrc` to append the lines output at the end of pyenv install.
- `exec $SHELL`
- `git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv`

Finally, setup the virtual env
- `pyenv install 3.7.7`
- `pyenv virtualenv 3.7.7 covid-data-model`
- `pyenv activate covid-data-model`
- `pip install --upgrade pip`
- `cd covid-data-model`
- `make setup-dev`

Something is broken with git-lfs. When using `git remote set-url origin https://github.com/covid-projections/covid-data-model` LFS can fetch files but to push back to the repo I used `git remote set-url origin git@github.com:covid-projections/covid-data-model` after doing something similar to https://github.com/covid-projections/can-scrapers/blob/main/services/prefect/README.md

Certbot setup, following https://certbot.eff.org/lets-encrypt/debianbuster-nginx
- `sudo apt install snapd`
- `sudo snap install hello-world`
- `sudo snap install --classic certbot`
- `sudo certbot --nginx`
... asks "Please enter in your domain name(s)" but I don't know.


At https://console.cloud.google.com/compute/instancesDetail/zones/us-west1-b/instances/data-pipeline-dashboard-1?project=covidactnow-dev&supportedpurview=project click Edit and enable HTTP and HTTPS.
Check that http://34.105.87.107/ connects to nginx.


### Configure and start uWSGI and nginx
- `pip install uwsgi`, don't `apt-get` because it gets a much older version.
- `make -C services/data-pipeline-dashboard setup

### Setup webhook


- `sudo apt-get install webhook`
- Edit services/webhook/secret.conf to add a secret
- `make -C services/webhook/ setup`
- Create a webhook on the github repo:
  - Set the payload URL to https://SERVER/webhook/pull-covid-data-model
  - Keep content type as `application/x-www-form-urlencoded`
  - Set the Secret field to the string you added in `secret.conf`
  - Only send `push` event
  - Make sure it is active
  - Submit

Problems?
- Look below `Recent Deliveries` at https://github.com/covid-projections/covid-data-model/settings/hooks/281997561 and `sudo systemctl status webhook` on the GCP instance to see how it is doing.
