from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
import tempfile

import base64
from io import BytesIO

import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import matplotlib as mpl


mpl.use('Agg')

from waveletAnalysis import plotWavelet


# https://medevel.com/flask-tutorial-upload-csv-file-and-insert-rows-into-the-database/

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
app.config['UPLOAD_FOLDER'] =  tempfile.mkdtemp()


# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')


# Get the uploaded files
@app.route("/", methods=['POST'])
def process():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # set the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # save the file
        uploaded_file.save(file_path)
    else:
        file_path = "static/files/sst_nino3.csv"
    # parse parameters
    params = {
            "transform": request.form.get("transform"),
            "max_power": int(request.form.get("max_power")),
            "scale1": int(request.form.get("scale1")),
            "scale2": int(request.form.get("scale2")),
            "levels": [0]+[float(l) for l in request.form.get("levels").split(",")]+[1e40],
            "dt_units": request.form.get("dt_units"),
            "units": request.form.get("units"),
            "title": request.form.get("title"),
            "cmap": request.form.get("cmap"),
            }
    #print(params)
    try:
        # Save it to a temporary buffer.
        fig = plotWavelet(file_path,params)
        buf = BytesIO()
        #params = {"title": request
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        img = f"<a href='data:image/png;base64,{data}'><img src='data:image/png;base64,{data}'/></a>"
        plt.close(fig)
        return render_template('data.html',  plot=img)
    except Exception as e:
        return render_template('error.html',exception=e)
    #else:
    #    return redirect(url_for('index'))

if __name__ == "__main__":
        app.run()
