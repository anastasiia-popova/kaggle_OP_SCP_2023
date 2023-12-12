FROM jupyter/base-notebook

ADD requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy files into the container
ADD scp-xgboost-in-compressed-space.ipynb .
ADD xgb_script.py .
ADD de_train.parquet .
ADD id_map.csv .

# Make port 8888 available to the world outside this container
# EXPOSE 8888

# Run 
# CMD ["jupyter", "notebook", "--port=8080", "--no-browser", "--ip=0.0.0.0", "--allow-root"]