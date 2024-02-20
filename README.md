# Student Exam Performance Prediction
HUDK4054 Individuals Assignment#2

# Project Information
* Title: Student Exam Performance Prediction
* Creator: Hannah Tang
* Data Manager: Hannah Tang
* ORCID ID: 0009-0000-8861-7053
* Affiliation: Teachers College, Columbia University
* Date: 20-02-2024

# Project Abstract
* This dataset is designed to support the prediction of exam outcomes (pass/fail) for students based on their study hours and previous exam scores. It's aimed at educators, researchers, and machine learning practitioners interested in understanding and forecasting academic performance.

# Collaborators and Authors
* Owner: Muslimbek Abdurakhimov
* Author: MrSimple07
* License: Apache 2.0
* DOI: 10.34740/kaggle/dsv/7623777

# Dataset Details
* Study Hours (numeric): Hours a student spent studying for the upcoming exam.
* Previous Exam Score (numeric): Student's score in the previous exam.
* Pass/Fail (binary): Target variable indicating exam outcome, where 1 signifies a pass and 0 signifies a fail.

# Size and Coverage
* Dataset Size: Data on 500 students.
* Temporal Coverage: From December 31, 2023, to January 13, 2024.
* Geospatial Coverage: Worldwide.

# Python Libraries
   * 1. KNN
X = df.drop('Pass/Fail', axis=1)
y = df['Pass/Fail']
add Codeadd Markdown
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
add Codeadd Markdown
scaler = StandardScaler()
knn = KNeighborsClassifier()
add Codeadd Markdown
operations = [('scaler', scaler),('knn', knn)]
pipe = Pipeline(operations)
add Codeadd Markdown
k_values = list(range(1,20))
param_grid = {'knn__n_neighbors': k_values}
cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
cv_classifier.fit(X_train, y_train)
add Codeadd Markdown
cv_classifier.best_estimator_.get_params()
add Codeadd Markdown
knn3 = KNeighborsClassifier(n_neighbors=3)
operations = [('scaler', scaler),('knn3', knn3)]
add Codeadd Markdown
pipe = Pipeline(operations)
add Codeadd Markdown
pipe.fit(X_train, y_train)
pipe_pred = pipe.predict(X_test)
add Codeadd Markdown
confusion_matrix(y_test, pipe_pred)
add Codeadd Markdown
print(classification_report(y_test, pipe_pred))
   * 2. EDA
df.info()
add Codeadd Markdown
df.describe().transpose()
add Codeadd Markdown
sns.countplot(x='Pass/Fail', data=df)
add Codeadd Markdown
sns.scatterplot(x='Previous Exam Score', y='Study Hours', data=df, hue='Pass/Fail')
plt.legend(bbox_to_anchor=(1.05, 1))
add Codeadd Markdown
sns.heatmap(df.corr(), cmap='viridis', annot=True)

# Data Storage
* Data will primarily be stored as a .csv file on the computers of the researchers.

# Final Observation
* Which metadata standard did you choose and why? I utilized a custom approach tailored to the dataset's specifics and user needs, rather than a formal metadata standard.This decision was based on the necessity for clarity, accessibility, and relevance to potential users from diverse backgrounds, including education, research, and data science.
* Which template/software did you use? it was crafted directly within this conversational interface based on the provided details about the dataset.
* What was the most challenging part of creating a ReadME file? How did you overcome these obstacles? The most challenging part of creating a README file often involves ensuring that the document is both comprehensive and accessible to a wide range of users, from experts to novices. This challenge is particularly pronounced when the dataset is complex or the intended use case is broad. To overcome these obstacles, it's crucial to focus on clear, I broke down complex concepts into simpler explanations and providing examples when possible can also help make the README more user-friendly. 
