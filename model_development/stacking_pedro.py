#### PLANNING STACKING ####
#(referencia a este site, vi bue coisas la) https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

'''
def get_stacking():
    # define the base models, level0 of the stack, tem aq exemplos
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))

    # define meta learner model, level1, final estimator
    level1 = LogisticRegression()

    # define the stacking ensemble, using scikit.learn
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


# para dar return de um dict com os modelos
def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    models['stacking'] = get_stacking()
    return models

# faz folds e avalia os modelos
    def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) # para fazer 
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def main():
    # definimos X,y 
    X = _
    y = _

    #pegamos nos modelos
    models = get_models()

    #-----------Avaliar a performance dos modelos em separado----------
    # evaluate the models and store results
    results, names = list(), list() 
    for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()

    # e importante para referencia ne
    #-------------------------------------------------------------------

    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    # fit the model on all available data
    model.fit(X, y)

    prediction = model.predict(data)
'''