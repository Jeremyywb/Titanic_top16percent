from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import collections as coll
import pandas as pd
import numpy as np
import re
import os



# import pdb
# pdb.set_trace()

os.chdir("D:/CODES/ProjectCT/Kaggle/TitanIC/")
# os.listdir("D:/CODES/ProjectCT/Kaggle/TitanIC/")

class dataprepare:
    """docstring for dataprepare"""
    def __init__(self, trainflie, testfile):
        self.trainflie = trainflie
        self.testfile = testfile

    def prepare(self):
        '''
           ****Age****
              maybe it's bad to create Age_None feature
              formula:data["Age_None"] = data.Age.isnull()+0
              with null data use mean  OR NEED FURTHER FEATURE enginering
           *Children*
              when age<=10 set children,but 18 was seen 
              on other kagger's blog, but I'm not sure
              will that  work, actually I don't know 
              wether to spitl by it or not at all
           ****More_rela****
              value = Parch + SibSp
              maybe try just use size ==2,3 to indicate
           ****Avg_Fare****
              replace missing with medain,then divided 
              Fare by Ticket count and scale it
              中位数可以使分布更加靠拢
              不过更科学的应该使用算法去填补缺失值，更加还原实际值
            ****categor_var****
              features to dummy:
                "Sex","SubName" ,"FirCabin", "Embarked","More_rela"
              some needs further consider:
                 "Pclass": 感觉存在趋势，暂时不用dummy
                 "Tic_count"
            ****drop some var****
              "Fare", "Name","Ticket","Cabin","Sex_male"
              why and why not:
                "Sex_male":is del after dummy:to make it single indicator
                'SibSp', 'Parch':not consider del,which is 
                                 add together as More_rela:just instinct
            
            ****questions left behind****
             1、don't know coding to mod defaul boostrap sampling 
               which use to sovle unblance(48% of survive) when I try
               to use oob data to estimate randomforest rather than 5-fold
               with nested cv=5 in gridsearch(5 times 5-fold crosvalidation)
             2、not sure the way I use cv is corret
               the way I did: split my train set into 5 split
               each time use a spilt as test others as train
                  ps:this is what k-fold do,it contains 5-pairs train-test
               and use gridsearch on "sub-train-set" with 
               nested cv=5 which just like to do another 
               crossvalidation in order to get best-performace-model
               (the best-performace-model is to test the out-side-test set)
               it's like chose the best from the best

        '''
        X = pd.read_csv(self.trainflie)
        Y= pd.read_csv(self.testfile)
        X["TrainID"] = 1
        Y["TrainID"] = 0
        Y["Survived"] = 99
        data = pd.concat([X,Y],axis =0)
        data.Age.fillna(data.Age.mean(),inplace=True)
        data["More_rela"] = data.Parch+data.SibSp
        #
        # data.More_rela = data.More_rela.map(lambda x:99 if x>3 else x)
        # data.Cabin.fillna("missing",inplace=True)
        data["FirCabin"] = data.Cabin.map(lambda v: v[0] if type(v) == str else "None")
        data["SubName"] = data.Name.map(dataprepare.findmrs)
        data["Children"] = data.Age.map(lambda x: 1 if x<=10 else 0)
        data["Fare_None"] = data.Fare.isnull()+0
        data.Fare.fillna(data.Fare.median(),inplace=True)
        data["Avg_Fare"] = preprocessing.scale(data.Fare/data.Tic_count)
        # data.Avg_Fare = preprocessing.scale(data.Avg_Fare)
        ###count for ticket
        tic_count = pd.DataFrame(data.Ticket.value_counts()).reset_index()
        tic_count.columns = ['Ticket', 'Tic_count']
        data = pd.merge(data,tic_count,on="Ticket",how="left")
        categor_var =["Sex","SubName" ,"FirCabin", "Embarked","More_rela"]    #
        for var in categor_var:
            print("what do you know, I'dont like it,deam ", var)
            data[var].fillna("missing",inplace = True)
            dummiesX = pd.get_dummies(data[var],prefix= var)
            data = data.join(dummiesX)
            data.drop([var], axis=1,inplace= True)

        data.drop( ["Fare", "Name","Ticket","Cabin","Sex_male"], 
                    axis=1,inplace= True)
        train_data = data[data.TrainID==1]
        test_data  = data[data.TrainID==0]
        del data
        del train_data["PassengerId"]
        del train_data["TrainID"]
        del test_data["TrainID"]
        del test_data["Survived"]
        Y_lable = train_data.pop("Survived") 
        ID_test = test_data.pop("PassengerId")
        return train_data, Y_lable,test_data,ID_test

    @staticmethod
    def findmrs(x):
        v = re.findall("Mrs|Miss|Mr|Master",x)
        if len(v)==0:
            return "Other"
        else:
            return v[0]

    def one_randomforest(self,
                        train_x, train_y, test_x,test_y,
                        grid_parameter, 
                        model_n,
                        inner_cv):##  cv = 5

        basemodel = RandomForestClassifier(max_features = 10,
                                           max_depth=20,
                                           oob_score=False, 
                                           n_jobs=-1,
                                           random_state=42)
        grid = GridSearchCV(estimator=basemodel, 
                            param_grid=grid_parameter,
                            cv=inner_cv)
        grid.fit(train_x, train_y)
        # grid.predict_proba()
        print("model %d inside best performers: "%model_n,grid.best_params_,"===>",grid.best_score_)
        y_true, y_scores = test_y, grid.best_estimator_.predict_proba(test_x)

         #'''
         #此处注意: y_true 是1维，同时y_scores 是 下面
         #                                    [0.98,0.02]
         #                                    [0.78,0.22]
         #    这样的情形，无法匹配，多多注意
         #  >>> y_true = np.array([0, 0, 1, 1])
         #  >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
         #  >>> roc_auc_score(y_true, y_scores)
         #'''
        return roc_auc_score(np.array([y_true,1-y_true]), 1-y_scores.T),grid.best_estimator_



if __name__ == '__main__':
    kfolds = 5  ##10 is worse then 5
    inner_cv = 5
    model_n = 0
    submitfile = "submition_14.csv"###chose a name

    grid_parameter = dict(n_estimators=[x for x in range(30,40,2)],
                          min_samples_leaf=[x for x in range(5,10,2)]
                         )
    datacentr = dataprepare("train.csv", "test.csv")
    X_data,Y_lable,test_data,test_ID = datacentr.prepare()

    print("shoot,shoot,shoot, here comes the model...")

    # chose_bestmodel = {}

    kf = KFold(n_splits = kfolds)
    control_left = coll.deque(maxlen=2)
    control_left.append(0)
    for train_index, test_index in kf.split(X_data.index):
        train_x,test_x = X_data.ix[train_index],X_data.ix[test_index]
        train_y,test_y = Y_lable[train_index],Y_lable[test_index]
        print(len(train_x))
        model_n = model_n+1
        ##这里return model的方式可能有错，当心
        one_model_score, condidate_model = datacentr.one_randomforest(train_x, 
                            train_y, test_x,test_y,
                            grid_parameter = grid_parameter, 
                            model_n =model_n,
                            inner_cv = inner_cv)
        if one_model_score>control_left[0]:
            control_left.append(one_model_score)
            control_left.append(condidate_model)

        # chose_bestmodel[str(model_n)] = [one_model_score, condidate_model]



        print("model %d testscore is: "%model_n, one_model_score)

    best_of_best = control_left.pop()
    chose = control_left.pop()
    print("the best model best scored by :%d"%chose,
        "the winner :",str(best_of_best))
   
    pretions = best_of_best.predict(test_data)
    submit = pd.DataFrame( list(zip(test_ID,pretions)))
    submit.columns = ["PassengerId","Survived"]
    submit.to_csv(submitfile,index=False)

    # import matplotlib.pyplot as plt
    # # x = [u'INFO', u'CUISINE', u'TYPE_OF_PLACE', u'DRINK', u'PLACE', u'MEAL_TIME', u'DISH', u'NEIGHBOURHOOD']
    # # y = [160, 167, 137, 18, 120, 36, 155, 130]
    # feature_importans = pd.Series(best_of_best.feature_importances_,index=X_data.columns)
    # feature_importans.sort()
    # figure(figsize=(8,10), dpi=200)
    # fig, ax = plt.subplots()    
    # # width = 0.75 # the width of the bars 
    # ind = np.arange(len(feature_importans.index))  # the x locations for the groups
    # ax.barh(ind, feature_importans, width, color="blue")
    # ax.set_yticks(ind+width/10)
    # ax.set_yticklabels(feature_importans.index, minor=False)
    # plt.title('title')
    # plt.xlabel('features')
    # plt.ylabel('auc')      
    # plt.show()
