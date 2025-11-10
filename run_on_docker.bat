:: If you need to use a network proxy during deployment, you can uncomment the lines below
:: and set the correct port. Note: Inside a Docker container, 'host.docker.internal'
:: points to the host machine's localhost (127.0.0.1).

:: set HTTP_PROXY=http://host.docker.internal:10808
:: set HTTPS_PROXY=http://host.docker.internal:10808

docker-compose down
docker-compose up --build -d

pause