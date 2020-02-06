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
	@echo "implement the starter for your application structure\n-> put the application in the source directory"
