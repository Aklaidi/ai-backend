AI Backend

In /ai-backend create venv

```
python3.11 -m venv venv
``` 

Activate venv

```
source venv/bin/activate
```

Install requirements

```
pip intall -r requirements.txt
```

Before running migrations make sure to run the docker compose to create the database

```
docker compose up -d
```

Run migrations

```
./manage.py migrate
```

Run server

```
./manage.py runserevr 8000
```

Inside the repository are also the models