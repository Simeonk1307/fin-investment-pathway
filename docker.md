# 🐳 **DOCKER CHEAT SHEET (Developer Edition)**

---

# 📦 **1. Images**

List images:

```
docker images
```

Pull an image:

```
docker pull ubuntu:latest
```

Remove an image:

```
docker rmi IMAGE
```

Build an image from Dockerfile:

```
docker build -t myapp .
```

---

# 🚢 **2. Containers**

Run a container:

```
docker run IMAGE
```

Run interactive shell:

```
docker run -it ubuntu bash
```

Run detached (background):

```
docker run -d IMAGE
```

Name a container:

```
docker run --name mycontainer IMAGE
```

Map ports (HOST:CONTAINER):

```
docker run -p 8080:80 IMAGE
```

Mount a file or folder (HOST:CONTAINER):

```
docker run -v $(pwd)/config.yaml:/etc/config.yaml IMAGE
```

Environment variables:

```
docker run -e KEY=value IMAGE
```

Stop:

```
docker stop NAME
```

Kill immediately:

```
docker kill NAME
```

Remove:

```
docker rm NAME
```

Remove *all stopped* containers:

```
docker rm $(docker ps -aq)
```

---

# 📋 **3. Inspecting**

Show running containers:

```
docker ps
```

Show all containers:

```
docker ps -a
```

Logs:

```
docker logs NAME
docker logs -f NAME    # follow
```

Info about container:

```
docker inspect NAME
```

---

# 🖥️ **4. Exec (Enter Running Container)**

Open shell inside container:

```
docker exec -it NAME bash
```

If no bash:

```
docker exec -it NAME sh
```

Run command inside container:

```
docker exec NAME ls /etc
```

---

# 🌐 **5. Networking**

List networks:

```
docker network ls
```

Create network:

```
docker network create mynet
```

Run container on network:

```
docker run --network=mynet IMAGE
```

Inside docker-compose, containers reach each other by service name.

---

# 💾 **6. Volumes**

List volumes:

```
docker volume ls
```

Create:

```
docker volume create pgdata
```

Use with container:

```
docker run -v pgdata:/var/lib/postgresql/data postgres
```

Remove volume:

```
docker volume rm pgdata
```

---

# 🧹 **7. Cleanup (Safe)**

Remove stopped containers:

```
docker container prune
```

Remove unused images:

```
docker image prune
```

Remove unused everything:

```
docker system prune
```

---

# 💀 **8. Cleanup (Dangerous)**

Remove ALL containers:

```
docker rm -f $(docker ps -aq)
```

Remove ALL images:

```
docker rmi -f $(docker images -q)
```

Remove EVERYTHING (containers, images, volumes, networks):

```
docker system prune -a --volumes
```

---

# 🧱 **9. Dockerfile Cheatsheet**

```dockerfile
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app code
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Command to run app
CMD ["python", "server.py"]
```

Build:

```
docker build -t myapp .
```

Run:

```
docker run -p 8000:8000 myapp
```

---

# 🧩 **10. Docker Compose (Most Useful Commands)**

Start all services:

```
docker compose up
```

Start in background:

```
docker compose up -d
```

Stop:

```
docker compose down
```

Rebuild after code change:

```
docker compose up --build
```

View logs:

```
docker compose logs -f
```
