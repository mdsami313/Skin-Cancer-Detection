from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import cv2
import PIL

app = Flask(__name__)
app.secret_key = "itissecret"
app.config['UPLOAD_FOLDER'] = "C:\\Users\\samis\\OneDrive\\Desktop\\Skin Cancer Detection\\static\\uploaded images"


extension_types = ["png", "jpg", "jpeg"]
file_ext = ("docx", "csv", "pdf", "xlsx", "txt")
classes = {4: ('nv', ' melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2 :('bkl', 'benign keratosis-like lesions'),
           1:('bcc' , ' basal cell carcinoma'),
           5: ('vasc', ' pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}

model = tf.keras.models.load_model("model/Skin Cancer.h5")

@app.route("/")
def home():
    return render_template("home.html", img_name=" ")

@app.route("/skin-prediction")
def route_to_prediction():
    return render_template('prediction.html', img_name=" ")

@app.route("/uploader", methods=["GET", "POST"])
def uploader():
    if request.method == "POST":
        path = str(request.files.get('ret-img'))
        print(path.split("'"))
        if path.split()[1] == "''":
            flash("Input Field Cannot be Empty!") 
            return redirect(url_for("uploader") + "#about")
        elif path.split("'")[1].endswith(file_ext):
            flash("Please Input An Image.")
            return redirect(url_for("uploader") + "#about")
        else:
            f = request.files['ret-img']
            img_name = f.filename
            print(img_name)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            print(img_name.split("."))
            print(f"static/uploaded images/{img_name}")

            img_path = f"static/uploaded images/{img_name}"
            image = PIL.Image.open(img_path)
            image = image.resize((28, 28))
            img_reshaped = np.array(image).reshape(-1, 28, 28, 3)
            pred = model.predict(img_reshaped)
            highest_acc = pred * 100
            accuracy = round(np.amax(highest_acc))
            # print(accuracy)
            pred_label = np.argmax(pred, 1)

            print(pred_label)
            disease = f"Detected: {classes[pred_label[0]][1]}."
            return render_template("prediction.html", img_name=img_name, prediction=disease, accuracy = accuracy)
            
    else:
        return render_template("prediction.html", img_name=" ")


if __name__=="__main__":
    app.run(debug=True)
