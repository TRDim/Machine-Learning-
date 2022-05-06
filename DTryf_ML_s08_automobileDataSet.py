# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# 1. Σύντομη παρουσίαση του dataset (τι περιγράφει).
# To dataset περιγράφει: 
# a) τον προσδιορισμο ενος αυτοκινήτου με βάση διαφόρων χαρακτηριστικών του, 
# b) την εκχωρημένη βαθμολογία ασφαλιστικού κινδύνου
# c) τις κανονικοποιημένες απώλειες κατά την χρήση σε σχέση με άλλα αυτοκίνητα 
# 
# Η δεύτερη βαθμολογία (b) αντιστοιχεί στον βαθμό στον οποίο το αυτοκίνητο είναι πιο επικίνδυνο από ό,τι υποδεικνύει η τιμή του. Στα αυτοκίνητα εκχωρείται αρχικά ένα σύμβολο παράγοντα κινδύνου που σχετίζεται με την τιμή του. Στη συνέχεια, εάν είναι πιο επικίνδυνο (ή λιγότερο), αυτό το σύμβολο προσαρμόζεται μετακινώντας το προς τα πάνω (ή προς τα κάτω) στην κλίμακα. Οι αναλογιστές αποκαλούν αυτή τη διαδικασία "συμβολισμό". Η τιμή +3 δείχνει ότι το αυτοκίνητο είναι επικίνδυνο, -3 ότι είναι πιθανώς αρκετά ασφαλές.
# 
# Οι παραπανω μεταβλητες αφορουν χαρακτηριστικα για ασφαλιστικες μελετες και δεν θα χρησιμοποιηθουν στην ανάλυσή μας. Θέλουμε να προβλέψουμε την τιμή με βάση τα υπόλοιπα χαρακτηριστικά 
# 
# 
# 2. Αριθμός δειγμάτων και χαρακτηριστικών, είδος χαρακτηριστικών. Υπάρχουν μη διατεταγμένα χαρακτηριστικά και ποια είναι αυτά;
# 
# #Δειγμάτων: 205
# #Χαρακτηριστικών: 26
# 
# Μη διατεταγμένα χαρακτηριστικά είναι  a) symboling  b) make c) fuel-type d)body-style e)drive wheels f)engine location g) fuel system 
# 
# 3. Υπάρχουν επικεφαλίδες; Αρίθμηση γραμμών;
# Οι επικεφαλίδες δεν υπήρχαν στο αρχικό dataset αλλά φορτώθηκαν απο το αρχειο names. Όσο για την αρίθμηση γραμμών υπάρχει το ανάλογο index 
# 
# 4. Ποια / ποιες είναι οι κολόνες με τις μεταβλητές - στόχους;
# H κολόνα με την μεταβλητή στόχο είναι αυτή της τιμής - 'price'
# 
# 5. Χρειάστηκε να κάνετε μετατροπές στα αρχεία text και ποιες?
# Mετατρέψαμε τo datatype για διάφορες μεταβλητές όπως φαίνεται στον αντίστοιχο πίνακα (πριν και μετα). Επίσης κανονικοποιούμε τα δεδομένα για να διευκολύνουμε τις συγκρίσεις. Τέλος αλλάζουμε 
# 
# 6. Υπάρχουν απουσιάζουσες τιμές; Πόσα είναι τα δείγματα με απουσιάζουσες τιμές και ποιο το ποσοστό τους επί του συνόλου;
# 
# **μεταβλητή / #δειγμάτων / %επι του συνόλου **
# 
# normalized-losses/41/20%
# 
# price/4/2%
# 
# stroke/4/2%
# 
# bore/4/2%
# 
# peak-rpm/2/1%
# 
# num-of-doors/2/1%
# 
# horsepower/2/1%
# 
# 
# 7. Διαχωρίστε σε train και test set. Εάν υπάρχουν απουσιάζουσες τιμές και μη διατεταγμένα χαρακτηριστικά διαχειριστείτε τα και αιτιολογήστε τις επιλογές σας.
# 
# Οι απουσιαζουσες τιμες αντικατασταθηκαν με τους μεσους ορους για τις μεταβλητες: normalized-losses, strokre, bore, peak-rpm, horsepower
# 
# Οι απουσιαζουσες τιμες για την μεταβλητη num-of-doors αντικατασταθηκε απο την πιο συνηθες τιμε 'four'
# 
# Οι απουσιαζουσες τιμες για την μεταβλητη price μας οδηγησαν να διαγραψουμε ολες τις εγγραφες μιας και ειναι η target μεταβλητη

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

# %% [code]
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv('../input/auto85/auto.csv', names = headers)
df.head()

# %% [markdown]
# In the following window we can observe the Variables and its datatype

# %% [code]
print(df.dtypes)


# %% [markdown]
# ### Dive into the data
# Replacing any missing data with NaN values

# %% [code]
df.replace('?', np.nan, inplace=True)
df.head()


# %% [code]
print('Shape of the Automobile Dataset:',df.shape)

# %% [code]
df.isnull().sum().sort_values(ascending = False).plot.bar(color = 'blue')
plt.title('Missing values per column', fontsize = 20);

# %% [code]
missing_data = df.isnull().sum()
missing_data.sort_values(inplace=True, ascending =False)
print( 'Columns with Missing Data:\n',missing_data)

# %% [code]
# Check dataTypes of the missing data Columns 
a=[]
print('Mean Values per missing Data Column:')
for i in range(len(missing_data)):
    if missing_data[i]!=0:
        a.append(missing_data.index[i])
        print(missing_data.index[i])

# %% [markdown]
# ### Replace Missing Values
# Since the dataset is not very large and the amount of missing data per column
# is not large compared with the column length we choose to replace the missing
# values with its **columns mean** instead of drop the whole record. For the 
# following variables: 
# * normalized-losses
# * stroke
# * bore
# * peak-rpm
# * horsepower 

# %% [code]
mean_norm_losses = df[a[0]].astype('float').mean(axis=0)
df['normalized-losses'].replace(np.nan, mean_norm_losses, inplace=True)

mean_norm_stroke = df[a[2]].astype('float').mean(axis=0)
df['stroke'].replace(np.nan, mean_norm_stroke, inplace=True)

mean_norm_bore = df[a[3]].astype('float').mean(axis=0)
df['bore'].replace(np.nan, mean_norm_bore, inplace = True)

mean_norm_peakRpm = df[a[4]].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, mean_norm_peakRpm, inplace=True)

mean_norm_horsepower = df[a[6]].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, mean_norm_horsepower, inplace=True)


# %% [markdown]
# * num-of-doors 
# 
# We have a string Value. Thus we replace 
# the NaN values with the most frequent value. 

# %% [code]
df['num-of-doors'].value_counts().idxmax() #Find value with max occurences

# %% [code]
df['num-of-doors'].replace(np.nan, 'four', inplace=True)

# %% [markdown]
# * price
# 
# For the missing values in the latter column we decide to remove 
# all the records without value since this is our prediction variable.

# %% [code]
df.dropna(subset=['price'], axis=0, inplace = True)
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# df.dtypes

# %% [markdown]
#  ### Change datatype format for several variables 

# %% [code]
print('Data Types prior of the changes:\n',df.dtypes)

# %% [code]
df[["peak-rpm", "price", "bore", "stroke"]] = df[["peak-rpm", "price", "bore", "stroke"]].astype("float")
df[["wheel-base", "length", "width", "height", "compression-ratio"]]=df[["wheel-base", "length", "width", "height", "compression-ratio"]].astype("float")
df[["curb-weight", "engine-size", "horsepower","city-mpg","normalized-losses" ]]  =df[["curb-weight", "engine-size", "horsepower","city-mpg","normalized-losses" ]].astype("int")



# %% [code]
print('Data Types after the changes:\n',df.dtypes)

# %% [code]
df2 = df.copy() #To create another DF in different memory position 

K = df2.values[0,:]
Vec = np.zeros_like(K)
for i in range(len(K)):
    #print(type(K[i]),type(K[i])== str )
    if type(K[i])== str:
        Vec[i]=1
        
print(Vec)

# %% [code]
# Encode the Categorical characteristics using the Label Encodes
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in range(0,len(Vec)):
    
    if Vec[i]==1:
        df2[df.columns[i]].values[:] = le.fit_transform(df2[df2.columns[i]])[:]

# %% [code]
df2

# %% [markdown]
# ### Aλλαγή των μεταβλητών City & Highway mpg σε km/h για καλύτερη κατανόηση 
# 
# ![image.png](attachment:image.png)

# %% [code]
df2['highway-mpg'] = 235/df2['highway-mpg']
df2.head()

# %% [code]
df2['city-mpg'] = 235/df2['city-mpg']
df2.head()

# %% [code]
df2 = df2.rename(columns={"highway-mpg":'highway-L/100km', "city-mpg": "city-L/100km"})

# %% [markdown]
# ### Data Normalization 
# 
# df- Initial Dataset
# 
# df2 - Dataset with Label Encoder
# 
# df3 - Dataset with Label Encoder & MinMaxScaler

# %% [code]
df3 = df2.copy()
Vec
Vec2 = (Vec!=1)
Vec2[-1]=False

# %% [code]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
df3[df3.columns[Vec2]] = scaler.fit_transform(df3[df3.columns[Vec2]])
#scaler.fit_transform(df3[df3.columns[1]])
#scaler.fit_transform(df3[df3.columns[i]])[:]


# %% [code]
df3.head()

# %% [code]
df2.head()

# %% [code]
# We want to plot the Correlation matrix to see which features are highly correlated and create new variables

import seaborn as sns

plt.figure(figsize=(12,12))
sns.heatmap(df2.corr(), annot=True)

# %% [markdown]
# ## Create Train - Test Set

# %% [code]
# Create X & Y for df2
X = df2[df2.columns[:-1]].values
Y = df2[df2.columns[-1]].values
# Split the Data 
dataset=0

from sklearn.model_selection import train_test_split

# Split our data
train, test, train_labels, test_labels = train_test_split(X, Y, test_size=0.2)

# %% [code]
# Create X & y for df3
X = df3[df3.columns[:-1]].values
Y = df3[df3.columns[-1]].values
# Split the Data 
dataset = 1
from sklearn.model_selection import train_test_split

# Split our data
train, test, train_labels, test_labels = train_test_split(X, Y, test_size=0.2)

# %% [code]
def plotLoss(test_labels, price_predict):
    x=np.arange(1,len(test_labels)+1,1)
    fig = plt.figure()
    fig.suptitle('Real_Price vs Pred_Price', fontsize=20)              # Plot heading 
    plt.xlabel('test', fontsize=18)                          # X-label
    plt.ylabel('predicted', fontsize=16) 
    plt.scatter(test_labels,price_predict, color= 'r')    
    #plt.plot(x,test_labels, color= 'g')
    #plt.scatter(x,np.abs(test_labels-price_predict), color= 'b')

    plt.show()


# %% [code]

def runModel(reg):
     
    reg =reg
    # Training the model
    reg.fit(train,train_labels)

    # Price prediction based on x_test data
    price_predict = reg.predict(test)
    price_pred = np.round(price_predict,2)
    price_pred_df = pd.DataFrame({'Predicted_price':price_pred})

    # Printing first 10 values of price prediction dataframe
    print(price_pred_df.head(10))

    # Accuracy percentage using y_test data
    # Here R^2 method is used to evaluate the model
    accuracy = r2_score(test_labels,price_predict)
    print()
    print("The accuracy of the model based on current test data: ",accuracy*100,"%")
    
    plotLoss(test_labels, price_predict)

    return accuracy*100 

# %% [code]
from collections import defaultdict
allScores = defaultdict(list)
allScores2 = defaultdict(list)
allScoresGS = defaultdict(list)

# %% [code]
# LINEAR MODELS
from sklearn import linear_model
from sklearn.metrics import r2_score
LinearModels =('LinearRegression', 'Ridge', 'Lasso')
allLinearModels=[]
allLinearModels.append(linear_model.LinearRegression())
allLinearModels.append(linear_model.Ridge(alpha=.5))
allLinearModels.append(linear_model.Lasso(alpha=0.1))


# %% [code]
from sklearn import linear_model
from sklearn.metrics import r2_score
import time 


print('Linear_Model\n')
all_Linear_Scores={}
for i in range(0,len(allLinearModels)):
    start_time= time.time()
    reg = allLinearModels[i]
    
    allScores[allLinearModels[i]].append(runModel(reg))
    timeS =time.time()-start_time
    allScores[allLinearModels[i]].append(timeS)
    
    #allScores

print(allScores.items())

# %% [code]
# GS LinearRegression()

from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  linear_model.LinearRegression()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]

# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
timeGS = time.time() - start_time
print('Time in seconds :%s seconds' %timeGS)

allScoresGS['LinearRegression'].append(result.best_score_*100)
allScoresGS['LinearRegression'].append(result.best_params_)
allScoresGS['LinearRegression'].append(timeGS)
print(allScoresGS)

# %% [code]
# GS Ridge()
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  linear_model.Ridge()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]

# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['Ridge'].append(result.best_score_*100)
allScoresGS['Ridge'].append(result.best_params_)
allScoresGS['Ridge'].append(timeGS)
print(allScoresGS)

# %% [code]
# GS linear_model.Lasso()
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  linear_model.Lasso()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['max_iter'] = [500, 750, 1000, 1500]
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]
space['selection'] = ['cyclic','random']

# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['Lasso'].append(result.best_score_*100)
allScoresGS['Lasso'].append(result.best_params_)
allScoresGS['Lasso'].append(timeGS)
print(allScoresGS)

# %% [code]


# %% [code]
# KERNEL RIDGE REG


from sklearn.kernel_ridge import KernelRidge
# sklearn.kernel_ridge.KernelRidge(alpha=1, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)

print('Kernel Ridge\n')
reg = KernelRidge(alpha=1.0)
start_time = time.time()

allScores['Kernel Ridge'].append(runModel(reg))
timeS =time.time()-start_time
allScores['Kernel Ridge'].append(timeS)
print(allScores)

# %% [code]
# GS KernelRidge()
from pandas import read_csv
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  KernelRidge()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['degree'] = [2, 3, 4, 5]
space['coef0'] = [1,2,3]

# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['KernelRidge'].append(result.best_score_*100)
allScoresGS['KernelRidge'].append(result.best_params_)
allScoresGS['KernelRidge'].append(timeGS)
print(allScoresGS)

# %% [code]


# %% [code]
# SGDRegressor
from sklearn.linear_model import SGDRegressor
#sklearn.linear_model.SGDRegressor(loss='squared_loss', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)

print('SGD Regressor\n')
reg = SGDRegressor(max_iter=1000, tol=1e-3)
start_time = time.time()


allScores['SGD Regressor'].append(runModel(reg))
timeS =time.time()-start_time
allScores['SGD Regressor'].append(timeS)
print(allScores)

# %% [code]
# GS SGDRegressor()
from pandas import read_csv
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  SGDRegressor()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['loss'] = [ 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
space['penalty'] = ['l2', 'l1', 'elasticnet']
space['max_iter'] = [500,750,1000,1200]




# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['SGDRegressor'].append(result.best_score_*100)
allScoresGS['SGDRegressor'].append(result.best_params_)
allScoresGS['SGDRegressor'].append(timeGS)
print(allScoresGS)

# %% [code]


# %% [code]
from sklearn.neighbors import KNeighborsRegressor

# sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)

print('KNeighborsRegressor \n')
reg = KNeighborsRegressor(n_neighbors=2)
start_time = time.time()

allScores['KNeighborsRegressor'].append(runModel(reg))
timeS = time.time() - start_time
allScores['KNeighborsRegressor'].append(timeS)
print(allScores)


# %% [code]
# GS KNeighborsRegressor()
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  KNeighborsRegressor()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['n_neighbors'] = [3,4,5,6,7]
space['weights'] = [ 'uniform', 'distance']
space['algorithm'] = ['ball_tree', 'kd_tree', 'brute']
space['leaf_size'] = [20,30,40]
space['p'] = [1,2]



# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['KNeighborsRegressor'].append(result.best_score_*100)
allScoresGS['KNeighborsRegressor'].append(result.best_params_)
allScoresGS['KNeighborsRegressor'].append(timeGS)
print(allScoresGS)

# %% [code]
from sklearn.tree import DecisionTreeRegressor

print('DecisionTreeRegressor\n')
#sklearn.tree.DecisionTreeRegressor(*, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0)
reg = DecisionTreeRegressor(max_depth=2)
start_time= time.time()

allScores['DecisionTreeRegressor'].append(runModel(reg))
timeS = time.time() - start_time
allScores['DecisionTreeRegressor'].append(timeS)
print(allScores)



# %% [code]
# GS DecisionTreeRegressor()
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]

model =  DecisionTreeRegressor()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['criterion'] = ['mse', 'friedman_mse', 'mae', 'poisson']
space['splitter'] = [ 'best', 'random']
space['min_samples_split'] = [2,3,4]



# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['DecisionTreeRegressor'].append(result.best_score_*100)
allScoresGS['DecisionTreeRegressor'].append(result.best_params_)
allScoresGS['DecisionTreeRegressor'].append(timeGS)
print(allScoresGS)

# %% [code]
# ENSEMBLE METHODS- Forests of Randomized Trees
# RandomForestReggressor 
from sklearn.ensemble import RandomForestClassifier

print('RandomForestClassifier \n')
reg =RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
start_time = time.time()

allScores['RandomForestClassifier'].append(runModel(reg))
timeS = time.time() - start_time
allScores['RandomForestClassifier'].append(timeS)
print(allScores)


# %% [code]
# GS RandomForestClassifier()
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]
y = y.astype('int')

model =  RandomForestClassifier()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['criterion'] = ['gini', 'entropy']
space['n_estimators'] = [80,100,200]
space['min_samples_split'] = [2,3,4]
space['max_features'] = ['auto','sqrt','log2']



# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['RandomForestClassifier'].append(result.best_score_*100)
allScoresGS['RandomForestClassifier'].append(result.best_params_)
allScoresGS['RandomForestClassifier'].append(timeGS)
print(allScoresGS)

# %% [code]
# ENSEMBLE METHODS - Forests of Randomized Trees
# ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

print('ExtraTreesClassifier\n ')
reg = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
start_time = time.time()

allScores['ExtraTreesClassifier'].append(runModel(reg))
timeS= time.time()-start_time
allScores['ExtraTreesClassifier'].append(timeS)

print(allScores)


# %% [code]
# GS ExtraTreesClassifier()
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import time 

start_time = time.time()

data = df3.values
X,y = data[:,:-1], data[:,-1]
y = y.astype('int')

model =  ExtraTreesClassifier()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

space = dict()
space['criterion'] = ['gini', 'entropy']
space['n_estimators'] = [80,100,200]
space['min_samples_split'] = [2,3,4]
space['max_features'] = ['auto','sqrt','log2']



# define search
search = GridSearchCV(model, space, scoring='r2', n_jobs=-1, cv=cv)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

timeGS = time.time()- start_time

allScoresGS['ExtraTreesClassifier'].append(result.best_score_*100)
allScoresGS['ExtraTreesClassifier'].append(result.best_params_)
allScoresGS['ExtraTreesClassifier'].append(timeGS)
print(allScoresGS)

# %% [code]
#BOOSTING 
#Voiting Regressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor


# Training classifiers
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

print('Voting Regressor \n GradientBoostingRegresor & RandomForestRegressor & LinearRegression \n')
reg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
start_time = time.time()


allScores['Voting Regressor'].append(runModel(reg))
timeS = time.time() - start_time
allScores['Voting Regressor'].append(timeS)

print(allScores)


# %% [code]
accGS=[]
timeGS=[]
infoGS=[]
for i in allScoresGS.keys():
    print(i)
    if i!=2:
        accGS.append(allScoresGS[i][0])
        infoGS.append(allScoresGS[i][1])
        timeGS.append(allScoresGS[i][2])

# %% [code]


# %% [code]
acc=[]
timee=[]
for i in allScores.keys():
    print(i)
    if i!= 'Voting Regressor':
        acc.append(allScores[i][0])
        timee.append(allScores[i][1])

# %% [code]
print('Best inputs for each model after GS:')
for i in range (0,len(infoGS)):
    print(labels[i])
    print(infoGS[i])
    

# %% [code]
len(accGS)

# %% [code]
ind = np.arange(0,18,2)
labels = ['LinearRegression','Ridge','Lasso','KernelRidge','SGDRegressor','KNeighborsRegressor','DecisionTreeRegressor','RandomForestClassifier','ExtraTreesClassifier']

plt.bar(ind, acc, width, label='df3')
#plt.bar(ind+width, acc2[:-1], width, label='df2 ') # Accuracy for the dataset without Scaler
plt.bar(ind-width, accGS, width, label='dfGS')

plt.ylabel('Scores %')
plt.title('Accuracy Scores for Each Used Model')


plt.xticks(ind  , labels , rotation=90)
plt.legend(loc='lower right')
plt.show()



# %% [code]
print('Train&Prefict in sec for GS: \n')
for i in range (len(timeGS)):
    
    print(labels[i],':', timeGS[i])

# %% [markdown]
# Παρατηρουμε πως για το κανονικοποιημένο dataset df3 τα accuracy ειναι καλυτερα.
# Επισης οι χρονοι είναι πολύ μικροτεροι για το για το train και fit στα μοντέλα χωρις GridSearch
# κατι που ηταν αναμενόμενο. Στα μοντέλα GridSearch δεν συμπεριλαβαμε  Scaler και LabelEncoder για τα κατηγορικα 
# δεδομλενα γεγονός που μάλλον εξηγεί την χαμηλότερη απόδοση των GS μοντέλων. 