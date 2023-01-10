# Inspired by https://github.com/act-now-coalition/can-scrapers/blob/main/services/prefect/Makefile

sync_nginx:
	sudo cp ./nginx.conf /etc/nginx/sites-available/data-pipeline-dashboard

sync: sync_nginx
	sudo cp ./uwsgi.service /etc/systemd/system/data-pipeline-dashboard.service

restart:
	sudo systemctl daemon-reload
	sudo systemctl restart data-pipeline-dashboard

setup_nginx: sync_nginx
	sudo rm -f /etc/nginx/sites-enabled/default
	sudo ln -s /etc/nginx/sites-available/data-pipeline-dashboard /etc/nginx/sites-enabled
	sudo systemctl restart nginx

setup: | setup_nginx restart
