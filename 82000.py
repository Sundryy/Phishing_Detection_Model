import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import re
from scipy.sparse import hstack, csr_matrix
from joblib import Parallel, delayed #Use all cores for RF


#reads csv dataset
df = pd.read_csv('phishing_email.csv')


def preparingSubjectBody():
    
    '''TF-IDF is name of the thing we need to use to prepare subject and body of each email'''




setOfSubjects = []
setOfBodies = []
setOfLabels = []


for i in enumerate(df.index):
    #print(i[0])
    
    #print(df['subject'][i[0]], i[0]  )

    #setOfBodies.append( df.['subject']      )

    #removed special characters + lowercased from subject
    cleanedSubject = re.sub(r"[^a-zA-Z0-9' ]", ' ', str(df['text_combined'][i[0]]).lower())
    cleanedSubject = " ".join(cleanedSubject.split()) #removes multi-spacing

    #cleanedBody = re.sub(r"[^a-zA-Z0-9' ]", ' ', str(df['body'][i[0]]).lower())
    #cleanedBody = " ".join(cleanedBody.split()) #removes multi-spacing
    

    setOfSubjects.append(cleanedSubject)
    #setOfBodies.append(cleanedBody)
    setOfLabels.append(int(df['label'][i[0]]))


print(len(setOfSubjects))
#print(setOfLabels)




#performs TF-IDF on subjects and bodies
tfidfSubject = TfidfVectorizer()
#tfidfBody = TfidfVectorizer()
subjectFeature = tfidfSubject.fit_transform(setOfSubjects)
#bodyFeature = tfidfBody.fit_transform(setOfBodies)





#url_sparse = csr_matrix(df['urls'].values).reshape(-1,1)
X = hstack([subjectFeature])


X_train, X_test, Y_train, Y_test = train_test_split(X, setOfLabels, test_size=0.2, random_state=42)

print("Total samples:", X.shape[0])
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])




def ModelRunningTest(X_train, X_test, Y_train, Y_test):
    print('Running RF model now....')
    #Random forest model using HOG extracted features
    #model = RandomForestClassifier(max_depth=50, n_estimators=1000, max_features='sqrt', n_jobs=-1, bootstrap=False, random_state = 10653054)
    #BETTERmodel = RandomForestClassifier(max_depth=100, n_estimators=850, max_features='sqrt', n_jobs=-1, bootstrap=False, random_state = 10653054)
    model = RandomForestClassifier(max_depth=150, n_estimators=500, max_features='sqrt', n_jobs=-1,bootstrap=False, random_state = 10653054)
    RF = model
    RF.fit(X_train, Y_train) # Training Classifier
    acc = RF.score(X_test, Y_test)

    print("Accuracy with Random Forset: {:.2f}%".format(acc * 100))


    RF_Prediction = RF.predict(X_test)
    print( '\nClasification report:\n', classification_report(Y_test, RF_Prediction))

    confusion_matrix = metrics.confusion_matrix(Y_test, RF_Prediction)
    display_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    display_matrix.plot()
    plt.show()



ModelRunningTest(X_train, X_test, Y_train, Y_test)



