import csv
import ast
from sklearn.feature_extraction import DictVectorizer
import numpy as np

def load_permissions():
    APK_FEATURES_PATH = ".\\amd_permissions.csv"
    reader = csv.DictReader(open(APK_FEATURES_PATH))
    results = []
    for row in reader:
        results.append(row)
    ANDROID_PERMISSIONS_PATH = ".\\AndroidPermissions.txt"
    with open(ANDROID_PERMISSIONS_PATH) as file:
        android_permissions = [line.strip().split('.')[-1] for line in file]
    #print("Android General Permissions list count: ", len(set(android_permissions)))

    print "APK Permissionss size: ", len(results)
    fam2counts = {}
    fam2noPerms = {}
    apk2permissions = {}
    no_permissions_count=0
    for row in results:
        permissions_dict = {}
#        if len(ast.literal_eval(row['permissions'])) > 0:
#            continue
        for permission in ast.literal_eval(row['permissions']):
            if permission.split('.')[-1] in android_permissions:  # Add valid permissions only
                permissions_dict[permission.split('.')[-1]] = 1

        if apk2permissions.has_key((row['family'], row['variety'], row['apk_name'])):
            print("Error - apk already exists")
            print(row['apk_name'])  # apk already exists
        else:
            fam2counts[row['family']] = fam2counts.get(row['family'], 0) + 1
            apk2permissions[(row['family'], row['variety'], row['apk_name'])] = permissions_dict
            if not permissions_dict:
                fam2noPerms[row['family']] = fam2noPerms.get(row['family'], 0) + 1
                no_permissions_count += 1
    print "apk with no permissions: ", fam2noPerms
    print "apk2permissions loaded count: ", fam2counts, "Total: " , len(apk2permissions.keys())
    return apk2permissions


def get_permissions(ids, labels, apk2perm):
    error_count = 0
    count = 0
    perm_list = []
    new_labels = []
    for id, label in zip(ids, labels):
        apk = Id2Apk[id]
        if apk2perm.has_key(apk):
            perm_list.append(apk2perm[apk])
            new_labels.append(label)
            count += 1
        else:
#            print apk
            perm_list.append({})
            new_labels.append(label)
            error_count += 1
    print ("Number of missing APKS: ", error_count)
    print ("Number of APKS: ", count)

    vec = DictVectorizer()

    perm_vec = vec.fit_transform(perm_list).toarray()
    print "Permissions Features count: ", len(vec.get_feature_names())

    print "Permissions Features Names: ", vec.get_feature_names()

    return perm_vec, np.array(new_labels)


CLASS_IDMAP_PATH = "D:\\AMData\\ComplexLab\\idMap.txt"
with open(CLASS_IDMAP_PATH) as f:
    Id2Apk = {int(line.strip().split(",")[0]) : (line.strip().split(",")[1], line.strip().split(",")[2], line.strip().split("\\")[-1]) for line in f}
f.close()