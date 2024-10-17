FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    doxygen \
    graphviz \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN doxygen Doxyfile

WORKDIR /app/docs/html

CMD ["python3", "-m", "http.server", "8000"]

EXPOSE 8000
