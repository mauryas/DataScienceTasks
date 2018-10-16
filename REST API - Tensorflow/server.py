#%% Import Libraries
import flask
from model.model import ImageClassifier
import io
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import zipfile
from PIL import Image
import psutil
import logging

from config.config import IMG_COLS, IMG_ROWS, TO_TRAIN

# Setting logging file path
try:
    log_path = os.stat(os.path.join(os.getcwd(), 'log'))
except:
    log_path = os.mkdir(os.path.join(os.getcwd(), 'log'))
    

# Logging object configrations
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='server.log',
                    filemode='w')

#%% Initializa the application and model
app = flask.Flask(__name__)
cls = None
#%% Methods 
def init_model():
    '''
    Initialize the model and train it for 
    '''
    
    global cls
    cls = ImageClassifier()
    if TO_TRAIN:
        cls.train()
        logging.info("Train Variable Value: {}". format(TO_TRAIN))
        logging.info("Straining to train model")
    else:
        cls.load_model()
        logging.info("Train Variable Value: {}". format(TO_TRAIN))
        logging.info("Loaded Trained Model")


#%% Post Methods
@app.route("/predict", methods=["POST"])
def predict():
    '''
    Read the input images from input and return the predicted class.
    '''
    
    result = {'success':False}
    #Fetch the file
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
             # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = np.reshape(image,(1,IMG_ROWS,IMG_COLS,1))

            y_pred = cls.predict(image)
            
            result['success'] = True
            
            result['prediction'] = (np.argmax(y_pred[0])).tolist()

    return flask.jsonify(result)

@app.route("/batch_train", methods=["POST"])
def batch_train():
    '''
    Take an input batch of images and train the classifier
    '''
    result = {'success':False}
    #Fetch the file
    if flask.request.method == "POST":
        """
	Validate the zip file which we will receive. If the available memory is 
        more than request file size + DNN model, then train the model. 
	"""
        avl_mem = (psutil.virtual_memory().free)*1024
        
        cls_mem = cls.get_model_memory_usage()
        content_length = flask.request.content_length
        logging.info('Mem: {}'.format(avl_mem - cls_mem -content_length))
        if (avl_mem - cls_mem -content_length) < 0.05*(avl_mem):
            result = {'success':False, 'Error': 'large file'}
            return flask.jsonify(result)
            
        
        if flask.request.files.get("zip"):
            zip_read = flask.request.files["zip"]
            logging.info('Zip read Sucess')
            # if the file uploaded is a zip, open the file and save create a numpy array
            zip_ref = zipfile.ZipFile(zip_read, 'r')
            # before extracting create a temp dir where we will save those files
            try:
                os.stat('tmp')
            except:
                os.mkdir('tmp')
            
            zip_ref.extractall(os.path.join(os.getcwd(), 'tmp'))
            zip_ref.close()
            new_train = []
            new_label = []
            
            # Now open the tmp dir and one by one extract the images 
            for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'tmp')):
                # we will get the labels from the file name 
                for file in files:
                    file_name = os.path.splitext(file)[0]
                    _label = file_name.split('_')[1]
                    # read the image and convert it into numpy array
                    image = Image.open(os.path.join(root,file))
                    image = np.reshape(image,(IMG_ROWS,IMG_COLS,1))
                    
                    # append the labels and image array to the list
                    new_train.append(image)
                    new_label.append(int(_label))
                    
            # One hot encoding
            enc = OneHotEncoder()
            new_labels = enc.fit_transform(np.reshape(new_label, (-1, 1))).toarray()
            # Now send this to the training
            logging.info(np.shape(new_train))        
            cls.batch_train_online(np.array(new_train), new_labels)
            logging.info('Batch Training is done.')
            result = {'success':True}

    return flask.jsonify(result)
    
    
if __name__ == "__main__":
    init_model()
    print('Model Loaded')
    app.run()

    
