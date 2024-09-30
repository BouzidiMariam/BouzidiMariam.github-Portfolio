
#!/usr/bin/env python
# coding: utf-8




from sklearn.calibration import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, precision_score, recall_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc





def load_data():
    data = pd.read_csv('metaverse_transactions_dataset.csv')
    return data

def preprocess_data(df):
    df=df.drop(['sending_address', 'receiving_address','ip_prefix'],axis=1)
    label_encoder = LabelEncoder()
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df['timestamp'].sort_values()
    df=df.sort_values('timestamp')
    df=df.reset_index()
    df=df.drop('index',axis=1)
    #df['transaction_type'].fillna('Unknown', inplace=True)
    #df['location_region'].fillna('Unknown', inplace=True)
    #df['anomaly'].fillna('Unknown', inplace=True)
    df['transaction_type_encoded'] = label_encoder.fit_transform(df['transaction_type'])
    df['location_region_encoded'] = label_encoder.fit_transform(df['location_region'])
    df["purchase_pattern_encoded"] = label_encoder.fit_transform(df["purchase_pattern"])
    df["age_group_encoded"] = label_encoder.fit_transform(df["age_group"])
    df['anomaly_encoded'] = label_encoder.fit_transform(df['anomaly'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df.drop(['timestamp', 'transaction_type', 'location_region', 'anomaly','age_group','purchase_pattern'], axis=1, inplace=True)
    df.dropna(subset=['anomaly_encoded'], inplace=True)
    return df





def split_data(df):
    df.dropna(subset=['anomaly_encoded'], inplace=True)
    y = df['anomaly_encoded']
    x = df.drop(columns=['anomaly_encoded'])
    x_train, x_test, y_train, y_test = train_test_split(x.copy(), y.copy(), test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test





def find_best_hyperparameters(model, params, x_train, y_train):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    return best_params





def plot_metrics(metrics_list, y_test, y_pred, class_names):
    unique_classes = np.unique(y_test)

    if 'Confusion Matrix' in metrics_list:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        st.pyplot()

    if 'Classification Report' in metrics_list:
        st.text("Classification Report:\n" + classification_report(y_test, y_pred, target_names=class_names))

    if 'ROC curve' in metrics_list:
        y_test_binarized = label_binarize(y_test, classes=unique_classes)
        y_pred_binarized = label_binarize(y_pred, classes=unique_classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(unique_classes)):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        for i in range(len(unique_classes)):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:0.2f}) for {unique_classes[i]}')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(len(unique_classes)):
            precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
            average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_binarized[:, i])

        plt.figure(figsize=(8, 6))
        for i in range(len(unique_classes)):
            plt.plot(recall[i], precision[i], label=f'Precision-Recall curve (AP = {average_precision[i]:0.2f}) for {unique_classes[i]}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        st.pyplot()









def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.title(' Metaverse ₿lockchain Transaction Risk Analysis')
    st.sidebar.title('Navigating Insights & Strategies 💹')

    df = load_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Data set")
        st.write(df)

    df = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ["low risk", "moderate risk","high risk"]
    
    if st.sidebar.checkbox("Show processed data", False):
        st.subheader("Processed data set")
        st.write(df)

    if st.sidebar.button("Compare Classifiers"):
        results = {}
        for Classifier in ["Support Vector Machine (SVM)", "Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors (KNN)", "Naive Bayes", "Neural Network"]:
            model = None
            if Classifier == "Support Vector Machine (SVM)":
                model = SVC()
                params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
            elif Classifier == "Logistic Regression":
                model = LogisticRegression()
                params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 200, 300, 400, 500]}
            elif Classifier == "Decision Tree":
                model = DecisionTreeClassifier()
                params = {'max_depth': range(1, 21), 'min_samples_split': range(2, 11), 'min_samples_leaf': range(1, 11)}
            elif Classifier == "Random Forest":
                model = RandomForestClassifier()
                params = {'n_estimators': range(100, 5001, 10), 'max_depth': range(100, 501)}
            elif Classifier == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier()
                params = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
            elif Classifier == "Naive Bayes":
                model = MultinomialNB()
                params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}
            elif Classifier == "Neural Network":
                model = MLPClassifier()
                params = {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)], 'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'solver': ['adam', 'sgd', 'lbfgs'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'learning_rate': ['constant', 'adaptive']}

            best_params = find_best_hyperparameters(model, params, x_train, y_train)
            model.set_params(**best_params)
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            results[Classifier] = accuracy

        best_classifier = max(results, key=results.get)
        worst_classifier = min(results, key=results.get)

        st.subheader("Best Classifier:")
        st.write(best_classifier)
        st.write("Accuracy:", round(results[best_classifier], 2))

        st.subheader("Worst Classifier:")
        st.write(worst_classifier)
        st.write("Accuracy:", round(results[worst_classifier], 2))

    else:
        st.sidebar.subheader("Choose Classifier")
        Classifier = st.sidebar.selectbox("Classifier", ["Support Vector Machine (SVM)", "Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors (KNN)", "Naive Bayes", "Neural Network"])
        best_params_button = st.sidebar.checkbox("Best Parameters", False)
        if Classifier == "Support Vector Machine (SVM)":
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
            gamma = st.sidebar.radio("Gamma(Kernel coefficient)",("scale","auto"),key="gamma")
            metrics = st.sidebar.multiselect('What metrics to plot?',('Confusion Matrix','ROC curve','Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Support Vector Machine Results")
                if best_params_button:
                    model = SVC()
                    params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
                    best_params = find_best_hyperparameters(model, params, x_train, y_train)
                    C = best_params['C']
                    kernel = best_params['kernel']
                    gamma = best_params['gamma']
                    st.write("Best Parameters:  C=", C ,"Kernel=",kernel,"gamma =",gamma)
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)

        elif Classifier == "Logistic Regression":
            C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
            max_iter = st.sidebar.slider("Maximum Number of iterations", 100, 500, key='max_iter')
            metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Logistic Regression Results")
                if best_params_button:
                   model = LogisticRegression()
                   params = {'C': [0.01, 0.1, 1, 10], 'max_iter': [100, 200, 300, 400, 500]}
                   best_params = find_best_hyperparameters(model, params, x_train, y_train)
                   C = best_params['C']
                   max_iter = best_params['max_iter']
                   st.write("Best Parameters:  C= ", C ,'  max_iter:  ',max_iter)
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)

                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)

        elif Classifier == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, key='max_depth')
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, key='min_samples_split')
            min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, key='min_samples_leaf')
            metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Decision Tree Results")
                if best_params_button:
                    model = DecisionTreeClassifier()
                    params = {'max_depth': range(1, 21), 'min_samples_split': range(2, 11), 'min_samples_leaf': range(1, 11)}
                    best_params = find_best_hyperparameters(model, params, x_train, y_train)
                    max_depth = best_params['max_depth']
                    min_samples_split = best_params['min_samples_split']
                    min_samples_leaf = best_params['min_samples_leaf']
                    st.write('Best Parameters:   max_depth: ',max_depth,'   min_samples_split:  ',min_samples_split,'   min_samples_leaf:  ',min_samples_leaf)
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)

        elif Classifier == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Estimators", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.slider("Max Depth", 100, 500, key='max_depth')
            metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Random Forest Results")
                if best_params_button:
                    model = RandomForestClassifier()
                    params = {'n_estimators': range(100, 5001, 10), 'max_depth': range(100, 501)}
                    best_params = find_best_hyperparameters(model, params, x_train, y_train)
                    n_estimators = best_params['n_estimators']
                    max_depth = best_params['max_depth']
                    st.write('Best Parameters:   max_depth: ',max_depth,'   n_estimators:  ',n_estimators)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(x_train, y_train)

                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)

        elif Classifier == "K-Nearest Neighbors (KNN)":
            n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, key='n_neighbors')
            weights = st.sidebar.radio("Weights", ("uniform", "distance"), key='weights')
            metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("K-Nearest Neighbors Results")
                if best_params_button:
                    model = KNeighborsClassifier()
                    params = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
                    best_params = find_best_hyperparameters(model, params, x_train, y_train)
                    n_neighbors = best_params['n_neighbors']
                    weights = best_params['weights']
                    st.write("Best Parameters: n_neighbors=  ",n_neighbors," weights=  " ,weights)
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                model.fit(x_train, y_train)

                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)

        elif Classifier == "Naive Bayes":
            alpha = st.sidebar.slider("Alpha (Smoothing Parameter)", 0.01, 2.0, step=0.01, key='alpha')
            fit_prior = st.sidebar.radio("Fit Prior", (True, False), key='fit_prior')
            metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Naive Bayes Results")
                model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                model.fit(x_train, y_train)

                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)

        elif Classifier == "Neural Network":
            hidden_layer_sizes = st.sidebar.select_slider("Hidden Layer Sizes", options=[(100,), (100, 100), (100, 100, 100)], key='hidden_layer_sizes')
            activation = st.sidebar.selectbox("Activation Function", ("identity", "logistic", "tanh", "relu"), key='activation')
            solver = st.sidebar.selectbox("Solver", ("adam", "sgd", "lbfgs"), key='solver')
            alpha = st.sidebar.slider("Alpha", 0.0001, 1.0, step=0.0001, key='alpha')
            learning_rate = st.sidebar.selectbox("Learning Rate", ("constant", "adaptive"), key='learning_rate')
            metrics = st.sidebar.multiselect('What metrics to plot?', ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Neural Network Results")
                if best_params_button:
                    model = MLPClassifier()
                    params = {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)], 'activation': ['identity', 'logistic', 'tanh', 'relu'],
                              'solver': ['adam', 'sgd', 'lbfgs'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'learning_rate': ['constant', 'adaptive']}
                    best_params = find_best_hyperparameters(model, params, x_train, y_train)
                    hidden_layer_sizes = best_params['hidden_layer_sizes']
                    activation = best_params['activation']
                    solver = best_params['solver']
                    alpha = best_params['alpha']
                    learning_rate = best_params['learning_rate']
                    st.write("Best parameters: activation=  ",activation, "  solver=  ", solver, "  alpha=  ",alpha, "  learning_rate=  ",learning_rate)
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate)
                model.fit(x_train, y_train)

                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, average='macro'), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, average='macro'), 2))
                plot_metrics(metrics, y_test, y_pred, class_names)





if __name__ == '__main__':
    main()



