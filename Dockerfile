# Utiliza una imagen base de Python
FROM python:3.11

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt para instalar las dependencias
COPY requirements.txt .

# Instala las dependencias usando pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia el contenido de la carpeta actual al directorio de trabajo en el contenedor
COPY . .

# Expone el puerto en el que se ejecutará la aplicación
EXPOSE 5000

# Ejecuta la aplicaci�n Python cuando se inicie el contenedor
CMD ["python", "app.py"]
