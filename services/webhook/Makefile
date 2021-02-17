setup-secret:
	! grep PUT_SECRET_HERE secret.conf
	sudo mkdir -p /etc/systemd/system/webhook.service.d
	sudo cp ./secret.conf /etc/systemd/system/webhook.service.d

sync:
	sudo cp ./webhook.service /etc/systemd/system/webhook.service


restart:
	sudo systemctl daemon-reload
	sudo systemctl restart webhook


setup: | setup-secret sync restart
	sudo systemctl status webhook
