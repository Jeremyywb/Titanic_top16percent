# Titanic_top16percent

一些特征解释（代码中有），以及遗留问题

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
