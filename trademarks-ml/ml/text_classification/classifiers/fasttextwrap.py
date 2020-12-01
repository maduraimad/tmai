from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import copy
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import fasttext
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pickle

class fasttextClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        # load the multilabelBinarizer from disk
        

    def load_object(self,filename):
        with open(filename, 'rb') as input:
            obj = pickle.load(input)
            print('file loaded: ' + str(filename))
        return obj

    def combine_fasttext(self,X,y):

        mlb = self.load_object("test_mlb.sav")
        
        
        labels = copy.deepcopy(y)

        #to get labels back from one hot encoding matrix
        labels = (mlb.inverse_transform(labels)) #creates list of label tuples
        labels = [list(elem) for elem in labels] #converts to lists of lists

        ident = str("__label__")

        #insert __label__ in front of the codes
        i=0
        for t in labels:
            j=0
            for a in t:
                labels[i][j] = ident + a
                j=j+1
            i=i+1

        #make all codes a single cell
        single_label = list(labels)

        i=0
        for t in labels:
            single_label[i] = ' '.join(single_label[i])
            i=i+1

        single_label_frame = pd.DataFrame()
        single_label_frame[['label']] = pd.DataFrame(single_label)

        description_frame = pd.DataFrame()
        description_frame[['text']] = pd.DataFrame(X)

        Xy_frame = pd.DataFrame()
        Xy_frame ['combined'] = single_label_frame ['label'] + ' ' + description_frame['text']

        #Xy = list(Xy_frame['combined'])
        return Xy_frame
        
        
    def fit(self, X, y):
        print('start of fit')
        # Check that X and y have correct shape
        ##X, y = check_X_y(X, y) #issue with text not being in int format
        
#         print ('y: '+str(y))
#         print ('ylength: '+str(len(y)))
#         print ('yshape: '+str(y.shape))
#         y = y.reshape(-1,1)
#         print ('y reshape: '+str(y))
#         print ('ylength reshape: '+str(len(y)))
#         print ('yshape reshape: '+str(y.shape))
        
        # Store the classes seen during fit
        print ('unique labels: ' + str(unique_labels(y)))
        self.classes_ = unique_labels(y)
        
        Xy = self.combine_fasttext(X,y)
        
        name="alpha"
        train_filename = str(name) + '_train.txt'
        model_filename = str(name) + '_model'

        Xy.to_csv(train_filename, sep=' ', index=False, header=False)
        
        print('training model')
        fasttext.supervised(train_filename, model_filename, loss='hs', silent=0)
        print ("model saved to: "+ str(model_filename))
        #print ("# labels identified: " + str(len(classifier.labels)))
       
        
        
        self.X_ = X
        self.y_ = y
    
        # Return the classifier
        return self

    def predict(self, X):
        
        mlb = self.load_object("test_mlb.sav")
        
        check_is_fitted(self, ['X_', 'y_']) # Check is fit had been called

        
        #X = check_array(X) # Input validation #issue with text not being int
       
        #make this an external variable or something later
        name="alpha"
        test_filename = str(name) + '_test.txt'
        model_filename = str(name) + '_model'
        
#         Xpandas = pd.DataFrame()
#         Xpandas [['text']] = pd.DataFrame(X) #save 
        
#         Xpandas.to_csv(test_filename, sep=' ', index=False, header=False) # save pandas dataframe to csv
        
#         subset_test = []
#         with open (test_filename, 'r') as lines:
#             for line in lines:   
#                 subset_test.append(line)
        
        print('loading model')
        ft_classifier = fasttext.load_model(str(model_filename) + '.bin') #Load pre-trained classifier
        
        flatX  = [val for sublist in X for val in sublist]
        print ('flattened X: ' + str(flatX [0:2]))
        
        flatX = [s + '\n' for s in flatX]
        print ('flatX w \n: ' + str(flatX [0:2]))
        
        atnum=5 #number of predictions to make for each test text
        print('predicting')
        result = ft_classifier.predict_proba(flatX, atnum)
        
        
        print('length of result after prediction:' + str(len(result)))
        print('result of prediction: ' + str(result[0:2]))
        print()
        
        print('removing __label__ from result')
        #remove __label__ from start of word
        i=0
        for t in result:
            j=0
            for a in t:
                result[i][j] = str(result[i][j][0])
                result[i][j] = result[i][j].strip('__label__')
                j=j+1
            i=i+1
        
        print('length of result after removing label:' + str(len(result)))
        print(result[0:2])
        print()
        
        
        print('transform result to one hot matrix')
        result= mlb.transform(result) #convert to one hot to return to sklearn
        
#         print('printing classes')
#         print(mlb.classes_)
        
        print('printing result and shape')
        print(result[0:2])
        print(result.shape)
        print('returning result')
        return result