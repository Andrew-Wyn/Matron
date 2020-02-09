GRADLE=./core/appw

all: requirements destroy compose

requirements:
	@echo '============ Checking if docker binaries exist...'
	@docker --version
	@docker-compose --version
	@echo '============ OK!'

compose:
	@echo '============ Creating docker environment...'
	docker-compose build --pull
	docker-compose up -d
	@echo '============ Docker environment for your project successfully created.'

destroy:
	@echo "============ Cleaning up docker environment..."
	docker-compose down -v
	docker-compose kill
	docker-compose rm -vf

start:
	docker-compose start

stop:
	docker-compose stop

app:
	@echo "executing the main script into app service..."
	docker-compose exec app python3 spammer.py

shell:
	@echo "opening shell into app service...\n\n\n ---- type 'exit' to quit ---- \n\n"
	docker-compose exec app /bin/bash