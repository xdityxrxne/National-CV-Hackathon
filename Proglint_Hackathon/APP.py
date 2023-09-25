from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic ={0: 'Glioma tumor', 1: 'Meningioma tumor', 2: 'No tumor', 3: 'Pituitary tumor'}

model = load_model('tumour.h5')

model.make_predict_function()

import numpy as np

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(150, 150))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 150, 150, 3)  # Make sure this matches your model's input shape
    class_probabilities = model.predict(i)
    predicted_class = np.argmax(class_probabilities, axis=1)[0]
    return dic[predicted_class]



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)