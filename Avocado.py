import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import pacf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import datetime

# In[Reading data]
def read_data():
    """
    Function to read data
    --------------------
    Outputs: 
        df     - dataframe
        xticks - monthly date ticks
    """
    df = pd.read_csv('avocado.csv', index_col=0)
    df.columns = (df.columns.str.lower()
                            .str.replace(' ','_')
                            .str.replace('averageprice','price'))
    df.rename(columns={'4046':'small_nr',
                       '4225':'large_nr',
                       '4770':'xlarge_nr',
                       'total_volume':'total_nr'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    if pd.unique(df['date'].dt.year == df['year']):
        df.drop(columns=['year'], inplace=True)
        
        df.sort_values(['date','region'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        xticks = df['date'].dt.strftime('%Y-%m').drop_duplicates()
    return df, xticks

# In[Insights]
def insights(df, xticks):
    """
    Function to provide some exploratory analysis
    ---------------------------------------------
    Inputs:
        df     - dataframe read by read_data()
        xticks - monthly date ticks
    Outputs:
        df - dataframe
    """
    def total_check(df, total, comp1, comp3, xticks):
        """
        Function to plot deviation of sum of individual components from their aggregate
        and either to remove an aggregate or introduce a new individual component.
        Deviation is plotted and saved.
        -------------------------------------------------------------------------------
        Inputs:
            df     - dataframe
            total  - aggregate
            comp1  - first component
            comp3  - third component
            xticks - monthly date ticks
        """
        plt.figure('Deviation_{}'.format(total), figsize=(12,6))
        plt.plot(df[total]-df.loc[:,comp1:comp3].sum(axis=1),
                 'o', markersize=1)
        plt.title('How {} deviate from its components'.format(total))
        plt.xticks(xticks.index, xticks,rotation=45)
        plt.savefig('Deviation_{}'.format(total))
        if abs(df[total]-df.loc[:,comp1:comp3].sum(axis=1)).max()<=1:
            df.drop(columns=[total], inplace=True)
        else: 
            df['other_'+total.rsplit('_')[1]] = df[total]-df.loc[:,comp1:comp3].sum(axis=1)
            df.drop(columns=[total], inplace=True)
    
    def count_frequency(df, aspect):
        """
        Function to count frequency of entries per aspect
        -------------------------------------------------
        Inputs:
            df     - dataframe
            aspect - aspect to count entries of
        Output:
            counts - frequency count
        """
        counts = pd.Series(df.groupby(aspect).count()
                             .groupby(df.columns[1])[df.columns[2]].count(),name='frequency')
        counts.index.name = 'entries_per_'+aspect
        return counts
    
    total_check(df,'total_bags','small_bags','xlarge_bags',xticks)
    total_check(df,'total_nr','small_nr','xlarge_nr',xticks)
    
    print(count_frequency(df,'date'),'\n')
    print(count_frequency(df,'region'),'\n')
    
    types = pd.Series(df.groupby('type').count()['date'], name='frequency')
    print(types,'\n')
    df.rename(columns={'type':'organic'}, inplace=True)
    df['organic'].replace({'organic':1, 'conventional':0}, inplace=True)
    
    plt.figure('Average_price', figsize=(12,6))
    plt.title('Average Avocado Price')
    plt.plot(df[(df['region']!='TotalUS') & (df['organic']==1)].groupby('region')['price'].mean(),'g-',label='average_price_organic')
    plt.plot(df[(df['region']!='TotalUS') & (df['organic']==0)].groupby('region')['price'].mean(),'b-',label='average_price_conventional')
    
    linspace = pd.unique(df['region']).shape[0]
    plt.plot(range(linspace),
             np.ones(linspace)*df[(df['region']=='TotalUS') & (df['organic']==1)].groupby('region')['price'].mean().values,
             label='US_average_organic',color='m')
    plt.plot(range(linspace),
             np.ones(linspace)*df[(df['region']=='TotalUS') & (df['organic']==0)].groupby('region')['price'].mean().values,
             label='US_average_conventional',color='r')
    plt.xticks(rotation=90, fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Average_price')
    
    df = df.reindex(columns=['date','region','small_nr','large_nr','xlarge_nr','other_nr',
                             'small_bags','large_bags','xlarge_bags','organic','price'])
    return df
# In[Preprocessing]
def split(X, y, region, columns, constant=False, trend=False, scale=False):
    """
    Function to split a dataset into train, validate, and test.
    It also allows to add a constant, trend, and scale observations.
    Automatically adds lags based on PACF.
    ---------------------------------------------------------------
    Inputs:
        X        - independet variables (np.ndarray)
        y        - dependent variable (np.array)
        region   - which region is being processed
        columns  - feature names
        constant - whether to add a constant (default False)
        trend    - whether to add a trend (default False)
        scale    - whether to scale features (default False)
    Outputs:
        X_train,y_train,X_validate,y_validate,x_test,y_test
        columns - feature names
    """
    if trend:
        X = np.concatenate([np.arange(1,X.shape[0]+1).reshape([-1,1]), X], axis=1)  
        columns = (columns[::-1]+['trend_{}'.format(region)])[::-1]
    if constant:
        X = np.concatenate([np.ones([X.shape[0],1]), X], axis=1)
        columns = (columns[::-1]+['constant_{}'.format(region)])[::-1]
       
    train = round(0.6*X.shape[0])
    validate = round(0.8*X.shape[0])
       
    lag=1
    for val in pacf(y[:train])[1:21]:        
        if val < -2/np.sqrt(y[:train].shape[0]) or val > 2/np.sqrt(y[:train].shape[0]):
            X = np.concatenate([X,shift(y.reshape([-1]),lag,cval=-999).reshape([-1,1])], axis=1)
            columns = (columns[::-1]+['lag_{0}_{1}'.format(lag,region)])[::-1]
            lag+=1
        else: 
            X = pd.DataFrame(X).replace(-999,np.nan).dropna().values
            y = y[lag-1:]
            break   
    
    if scale:
        X = StandardScaler().fit_transform(X)
        
    X_train = X[:train]
    X_validate = X[train:validate]
    X_test = X[validate:]
    
    y_train = y[:train]
    y_validate = y[train:validate]
    y_test = y[validate:]    
    return X_train,y_train, X_validate,y_validate, X_test,y_test,columns

def block(X, X_new, y, y_new):
    """
    Function to create a sparse matrix with distinct regions on its diagonal
    and to extend a dependent feature
    ------------------------------------------------------------------------
    Inputs:
        X     - matrix (independent features)
        X_new - matrix to combine
        y     - vector (dependent feature)
        y_new - vector to combine
    Outputs:
        X - sparse matrix
        y - extended dependent feature
    """
    X = np.block([[X, np.zeros([X.shape[0],X_new.shape[1]])],
                      [np.zeros([X_new.shape[0],X.shape[1]]), X_new]])
    y = np.concatenate([y,y_new],axis=0)    
    return X,y

def variables(df,scale):
    """
    Function to create create sparse matrices and extend dependent features
    for train, validate, and test data. 
    All regions are combined into mentioned matrices.
    -----------------------------------------------------------------------
    Inputs:
        df    - dataframe
        scale - whether dependent features should be scaled
    Outputs:
        X_train,y_train,X_validate,y_validate,x_test,y_test
        columns - feature names        
    """
    df1 = df.drop(columns=['price','date','organic'])
    i=0
    for region in pd.unique(df1['region']):              
        if i==0:
            X = df1[df1['region']==region].drop(columns=['region'])
            columns = [col+'_'+region for col in X.columns]
            X = X.values
            y = df[df['region']==region]['price'].values.reshape([-1,1])
            
            X_train,y_train, X_validate,y_validate, X_test,y_test,columns = split(X,y,region,columns,scale)
            
        else:
            X_new = df1[df1['region']==region].drop(columns=['region'])
            columns_new = [col+'_'+region for col in X_new.columns]
            X_new = X_new.values
            y_new = df[df['region']==region]['price'].values.reshape([-1,1])        
            
            X_train_new,y_train_new, X_validate_new,y_validate_new,X_test_new,y_test_new,columns_new = split(X_new,y_new,region,columns_new,scale)
            
            X_train,y_train = block(X_train,X_train_new,y_train,y_train_new)
            X_validate,y_validate = block(X_validate,X_validate_new,y_validate,y_validate_new)
            X_test,y_test = block(X_test,X_test_new,y_test,y_test_new)
            columns.extend(columns_new)
        i+=1
    columns = pd.Series(columns, name='feature')
    return X_train,y_train, X_validate,y_validate, X_test,y_test,columns

def model(X_train,y_train, X_test,y_test, n):
    """
    Function to perform optimization with Gradient Boosting Regressor
    Inputs:
        X_train - independent train data
        y_train - dependent train data
        X_test  - independent test data
        y_test  - dependent test data
        n       - number of estimators used
    Outputs:
        estimate           - predicted y_train
        predict            - predicted y_test
        feature_importance - feature importance of the classification (pd.Series)
    """
    gbr = GradientBoostingRegressor(n_estimators=n).fit(X_train,y_train.reshape([-1]))
    estimate = gbr.predict(X_train)
    predict = gbr.predict(X_test)
    feature_importance = pd.Series(gbr.feature_importances_, name='importance')
    return estimate,predict,feature_importance

# In[Model Evaluation] search==True ---> ~ 1 hours
def estimators_model(df, n_estimators=None, search=True, validate=True, scale=False):
    """
    Function to find an optimal number of estimators or evaluate chosen parameters
    ------------------------------------------------------------------------------
    Inputs:
        df           - dataframe
        n_estimators - number of estimators used in Gradient Boosting Regressor (default None)
        search       - whether to search of an optimal number of estimators (default True)
        validate     - whether to use a validate or test dataset (default True)
        scale        - whether to scale independent features
    Outputs:
        if search == True:
            train_list - list of R-squared's for train data
            test_list  - list of R-squared's for test data
        else:
            train           - R-squared's for train data
            test            - R-squared's for test data
            features_org    - feature importance for organic (pd.DataFrame)
            features_nonorg - feature importance for conventional (pd.DataFrame)
    """
    def evaluate(df, n_estimators, validate, scale):
        """
        Function to evaluate model's performance
        ----------------------------------------
        Inputs:
            df           - dataframe
            n_estimators - number of estimators
            validate     - whether to use validate or test data
            scale        - whether to scale independent features
        Outputs:
            y_train
            estimate        - predicted y_train
            y_test
            predict         - predicted y_test
            features_org    - feature importance for organic (pd.DataFrame)
            features_nonorg - feature importance for conventional (pd.DataFrame)            
        """
        print('Nr_estimators: ', n_estimators)
        start = datetime.datetime.now()
        
        if validate:
            X_train_org,y_train_org, X_test_org,y_test_org, _,_, columns_org = variables(df[df['organic']==1],scale)
            X_train_nonorg,y_train_nonorg, X_test_nonorg,y_test_nonorg, _,_, columns_nonorg = variables(df[df['organic']==0],scale)
        else:
            X_train_org,y_train_org, X_val_org,y_val_org, X_test_org,y_test_org, columns_org = variables(df[df['organic']==1],scale)
            X_test_org = np.concatenate([X_val_org,X_test_org],axis=0)
            y_test_org = np.concatenate([y_val_org,y_test_org],axis=0)
            X_train_nonorg,y_train_nonorg, X_val_nonorg,y_val_nonorg ,X_test_nonorg,y_test_nonorg, columns_nonorg = variables(df[df['organic']==0],scale)
            X_test_nonorg = np.concatenate([X_val_nonorg,X_test_nonorg],axis=0)
            y_test_nonorg = np.concatenate([y_val_nonorg,y_test_nonorg],axis=0)
       
        estimate_org, predict_org,feature_importance_org = model(X_train_org,y_train_org,X_test_org,y_test_org,n_estimators)
        features_org = pd.DataFrame(pd.concat([columns_org,feature_importance_org],axis=1))
        features_org = features_org[features_org['importance']!=0].reset_index(drop=True)
        features_org['region'] = pd.DataFrame(features_org['feature'].str.rsplit('_').tolist()).iloc[:,-1]
        features_org['relative_importance'] = features_org.groupby('region')['importance'].apply(lambda x: x/x.sum())
        
        estimate_nonorg, predict_nonorg,feature_importance_nonorg = model(X_train_nonorg,y_train_nonorg,X_test_nonorg,y_test_nonorg,n_estimators)
        features_nonorg = pd.DataFrame(pd.concat([columns_nonorg,feature_importance_nonorg],axis=1))
        features_nonorg = features_nonorg[features_nonorg['importance']!=0].reset_index(drop=True)
        features_nonorg['region'] = pd.DataFrame(features_nonorg['feature'].str.rsplit('_').tolist()).iloc[:,-1]
        features_nonorg['relative_importance'] = features_nonorg.groupby('region')['importance'].apply(lambda x: x/x.sum())
        
        y_train = np.concatenate([y_train_org,y_train_nonorg],axis=0)
        estimate = np.concatenate([estimate_org,estimate_nonorg],axis=0)
            
        y_test = np.concatenate([y_test_org,y_test_nonorg],axis=0)
        predict = np.concatenate([predict_org,predict_nonorg],axis=0)
        
        print('Evaluation time: ',datetime.datetime.now()-start)
        return y_train,estimate, y_test,predict, features_org,features_nonorg
    
    if validate==False and search:
        search=False
    
    if search:
        train_list=[]
        test_list=[]
        for estimators in np.arange(100,2000,100):
            y_train,estimate, y_test,predict, _,_ = evaluate(df,estimators,validate,scale)
            r2_train = r2_score(y_train,estimate)
            train_list.append(r2_train)
            print('R-squared train: {}'.format(round(r2_train,4)))
            r2_test = r2_score(y_test,predict)
            test_list.append(r2_test)
            print('R-squared validate: {}\n'.format(round(r2_test,4)))
        return train_list, test_list
    else:
        if n_estimators==None:
            raise ValueError("n_estimators must be int if search is False")
        y_train,estimate, y_test,predict, features_org,features_nonorg = evaluate(df,n_estimators,validate,scale)
        train = r2_score(y_train,estimate)
        test = r2_score(y_test,predict)
        return train, test, features_org, features_nonorg

# In[Plot Model Evaluation]
def plot_model(train, test, validate=True):
    """
    Function to plot and save model performance on train and test data
    ------------------------------------------------------------------
    Inputs:
        train    - list of R-squared's for train data
        test     - list of R-squared's for test data
        validate - whether to use validate data as test data (default True)
    Outputs:
        n_estimators - number of optimal estimators
    """
    if validate: label='Validate'
    else: label='Test'
    plt.figure('Train_{}'.format(label),figsize=(12,6))
    plt.plot(train,label='Train data',color='b')
    plt.plot(test,label=label+' data',color='green')
    max_idx = test.index(max(test))
    plt.vlines(x=max_idx, ymin=0.95*min(test), ymax=1.04*train[max_idx],
               colors='r',linestyle=':')
    plt.text(x=max_idx-1,y=1.05*train[max_idx],s='max R-squared',color='r')
    plt.text(x=max_idx,y=0.98*train[max_idx],s='{}'.format(round(train[max_idx],4)))
    plt.text(x=max_idx,y=1.01*test[max_idx],s='{}'.format(round(test[max_idx],4)))
    plt.xticks(np.arange(0,20,1),np.arange(100,2000,100), rotation=90)
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Number of estimators')
    plt.ylabel('R-squared')
    plt.ylim([0.9*min(test),1.1])
    plt.xlim([0,19])
    plt.savefig('Evaluation_graph')
    n_estimators = (max_idx+1)*100
    return n_estimators

# In[MAIN]
if __name__=='__main__':
    print('Reading data...')    
    df,xticks = read_data()
    df = insights(df,xticks)
    print('Model is being evaluated for different number of estimators...')
    train_list,validate_list = estimators_model(df)
    print('Evaluation graph is saved as Evaluation_graph.png...')
    n_estimators = plot_model(train_list,validate_list)
    print('Performance on the test data...')
    train,test,features_org,features_nonorg = estimators_model(df,n_estimators=n_estimators,search=False,validate=False,scale=False)
    print('Organic feature importance is saved as features_org.csv')
    features_org.to_csv('features_org.csv')
    print('Conventional feature importance is saved as features_conv.csv')
    features_nonorg.to_csv('features_conv.csv')
    print()
    print('Train R-squared: {}'.format(train))
    print('Test R-squared: {}'.format(test))




        



