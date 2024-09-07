FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
    
RUN pip3 install --no-cache-dir -r /app/requirements.txt
RUN pip3 install nltk
RUN python3 -m nltk.downloader punkt
RUN chmod -R 777 /usr/local/lib/python3.10/site-packages/llama_index/core/_static/nltk_cache/

# User
RUN useradd -m -u 1000 user
USER user
ENV HOME /home/user
ENV PATH $HOME/.local/bin:$PATH

WORKDIR $HOME
RUN mkdir app
WORKDIR $HOME/app
COPY . $HOME/app

EXPOSE 8501
CMD streamlit run app.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none