# Inspired by https://github.com/covid-projections/can-scrapers/blob/main/services/prefect/Makefile


sync:
	sudo cp ./uwsgi.service /etc/systemd/system/data-pipeline-dashboard.service
	sudo cp ./nginx.conf /etc/nginx/sites-available/data-pipeline-dashboard

restart:
	sudo systemctl daemon-reload
	sudo systemctl restart data-pipeline-dashboard

setup_nginx:
	sudo rm -f /etc/nginx/sites-enabled/default
	sudo ln -s /etc/nginx/sites-available/data-pipeline-dashboard /etc/nginx/sites-enabled
	sudo systemctl restart nginx
