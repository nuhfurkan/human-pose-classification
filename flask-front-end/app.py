from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from serverfiles.mpprocess import MPObject

app = Flask(__name__)
app.config["SECURE_KEY"] = "1234"
app.config['UPLOAD_FOLDER'] = 'static/files'

myTest = MPObject()

@app.route('/upload', methods=['POST'])
def upload():
    if 'images' in request.files:
        images = request.files.getlist('images')
        
        # Process the uploaded images as needed
        for image in images:
            # Save the image or perform any desired operations
            image.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(image.filename+".png"))) # Then save the file
            pass

        # here check the position
        # send the results as feedback        
        return str(myTest.fetchResults(app.config["UPLOAD_FOLDER"]+"/imagename.png"))
    return "no feedback"

@app.route('/', methods=['GET',"POST"])
def index():
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True, port=3000)