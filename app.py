import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = load_model(os.path.join(BASE_DIR , 'alphabet_recognition2.hdf5'))
model = load_model(os.path.join(BASE_DIR , 'model_1.hdf5'))
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename , model):

    img = load_img(filename , target_size = (32 , 32))
    img = img_to_array(img)
    img = img.reshape(1, 32 ,32 ,3)
    img = img.astype('float32')
    img = img/255.0

    result = model.predict(img)
    result_classes = result.argmax(axis=-1)
    # label_map = (result_classes.class_indices)
    # result_classes2 = chr(result_classes)

    prob = (result[0]*100)
    prob.sort()
    prob = prob[-1]
    return result_classes , prob

@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename
                class_result , prob_result = predict(img_path , model)
                predictions = {
                      "class1":class_result,
                      "prob1": prob_result,
                }
            except Exception as e : 
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'
            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error) 
            
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                class_result , prob_result = predict(img_path , model)
                predictions = {
                      "class1":class_result,
                      "prob1": prob_result,
                }
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug = True)