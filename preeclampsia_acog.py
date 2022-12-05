"""

"""
from json.encoder import py_encode_basestring
import pdb
import pandas as pd


def acog_test(mom):
    """
    """
    if (mom['Multi_Birth_Indicator'] == 1 or mom['Hypertension_PMH'] == 1 or mom['Pregestational_Hypertension_Indicator'] == 1 or
        mom['PreGestational_Diabetes_Indicator'] == 1 or mom['Pre_Eclampsia_PMH'] == 1 or mom['Diabetes_PMH'] == 1 or
        mom['CKD_PMH'] == 1 or mom['Kidney_Disease_Indicator'] == 1):
        acog_rating = 2
        return acog_rating
    else:
        acog_rating = 0
    
    mod_num = 0
    # Need example data
    if mom['Parity'] == 0:
        mod_num+=1
    if mom['Race_description'] == 'Black':
        mod_num+=1
    if mom['Medicaid_Indicator'] == "Y":
        mod_num+=1
    if mom['Maternal_Age'] > 35:
        mod_num+=1
    if mom['Pre_Pregnancy_BMI'] > 30:
        mod_num+=1
    if mod_num > 1:
        acog_rating = 1
    return acog_rating

    # res = table(mom$ACOG, mom$Pre_Eclampsia_Indicator)
    # sensitivity = res[2,2]/(res[2,2] + res[1,2])
    # specificity = res[1,1]/(res[1,1] + res[2,1])
    # ppv = res[2,2]/(res[2,2] + res[2,1])
    # npv = res[1,1]/(res[1,1] + res[1,2])
    # accu = (res[1,1] + res[2,2])/sum(res)
    # lr = (res[2,2]/res[2,1])/(res[1,2]/res[1,1])
    # post.prob = res[2,2]/(res[2,2] + res[2,1])


def check_acog(file):
    mom = pd.read_csv(file)
    mom['acog_rating'] = mom.apply(lambda row: acog_test(row), axis=1)
    mom['acog_check'] = mom['acog_rating'] > 0

    pdb.set_trace()
    mod_num = 0




if __name__ == '__main__':
    FILE = "xxxx.txt"
    check_acog("data/PatientDataSample.csv")