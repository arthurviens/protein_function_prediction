###############################################################################
# Submission script helper
#
# the goal of this file is to provide participants with the basic structure of
# the reading and writing of the datasets, as well as saving the results for
# submission.
#
#
# Participants need to submit a result (prediction) file that will be scored
# against the ground truth. Participants have to make a zip file (no constrain
# on the namefile), with your results as a vector matrix inside a txt file
# named `results.txt`.
#
# This file has greatly been inspired from the R submission script from Magali
# Richard and Florian Chuffard

import numpy as np
from zipfile import ZipFile
from sklearn import linear_model
from datetime import datetime

debut = datetime.now()

X = np.loadtxt("data/protein_train.data")
y = np.loadtxt("data/protein_train.solution")

X_test = np.loadtxt("data/protein_test.data")
X_valid = np.loadtxt("data/protein_valid.data")

###############################################################################
# Fit and predict

log_reg = linear_model.LogisticRegression()
log_reg.fit(X, y)

# Predict on the test and validation data.
y_test = log_reg.predict(X_test)
y_valid = log_reg.predict(X_valid)

###############################################################################
# Save results

np.savetxt("protein_test.predict", y_test, fmt="%d")
np.savetxt("protein_valid.predict", y_valid, fmt="%d")

zip_obj = ZipFile('submission.zip', 'w')
zip_obj.write("protein_test.predict")
zip_obj.write("protein_valid.predict")

zip_obj.close()

###############################################################################
# How to submit the zip file?
#
# Submit the zip submission file on the challenge in the tap "Participate,"
# menu "Submit / View results menu", sub-menu "Challenge #1" by clicking on
# the submit button after filling out some matadate.
#
# On the codalab challenge web page, The *STATUS* become :
#   - Submitting
#   - Submitted
#   - Running
#   - Finished
#
# When it’s finished :
#   - You refresh the page and see your score
#   - If enable, details for report could be downloaded by clicking *Download
#     output from scoring step*.
#   - Some logs are available to download.
#   - Leader board is updated in the `Results` tab.

fin = datetime.now()

print(f"Début {debut} and fin {fin}")
print(f"Time spent : {fin - debut}")