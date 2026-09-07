FROM osgeo/gdal:ubuntu-small-3.6.2

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs dem_catalog uploads/dem_cache

EXPOSE 8000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
