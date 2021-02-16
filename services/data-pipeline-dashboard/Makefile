# Inspired by https://github.com/covid-projections/can-scrapers/blob/main/services/prefect/Makefile


sync:
	sudo cp services/data-pipeline-dashboard/uwsgi.service /etc/systemd/system/data-pipeline-dashboard
	sudo cp services/data-pipeline-dashboard/nginx.conf /etc/nginx/sites-available/data-pipeline-dashboard

restart:
	sudo systemctl daemon-reload
	sudo systemctl restart data-pipeline-dashboard

setup_nginx:
	sudo rm /etc/nginx/sites-enabled/default
	sudo ln -s /etc/nginx/sites-available/data-pipeline-dashboard /etc/nginx/sites-enabled
