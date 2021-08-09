from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# define a SVC classifier
clf = SVC()

# define the class encodings and reverse encodings
classes = {0: "No Bugs", 1: "Bug"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the model
    # Load from file
    pkl_filename = './notebooks/svm_clf.pkl'
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = pickle_model.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.bug_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)