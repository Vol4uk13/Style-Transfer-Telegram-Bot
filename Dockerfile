# syntax=docker/dockerfile:2

FROM python:3.12
COPY . /app
WORKDIR /app
# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-di
<<<<<<< HEAD
CMD ["python", "main_for_many_users.py"]
=======
CMD ["python", "main.py"]
>>>>>>> eca0a41b6b4fc54d2daf2763be84f3b52653ec54
