## {{{ http://code.activestate.com/recipes/576755/ (r3)
# based on Recipe 466302: Sorting big files the Python 2.4 way
# by Nicolas Lehuen

import os
from os import remove, rename
from os.path import exists
from tempfile import gettempdir
from itertools import islice, cycle
from collections import namedtuple
from ConstantDefinitions import APK_COUNT
import csv
import heapq
from collections import Counter
from tempfile import NamedTemporaryFile

from ConstantDefinitions import STORAGE_PATH, UTILITY_PATH, CHUNK_SIZE
Keyed = namedtuple("Keyed", ["key", "obj"])
import re

MODES = ["BINARY", "TF", "TFIDF", "TF_NORM", "TF_NORM_IDF"]


def memory():
    import os
    from wmi import WMI
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)


def get_words_from_file(path, delimiter=r"[^\W_']+"):
    """
    Get all of the words in a file in a list delimited by space
    :param path: Path to the file
    :param delimiter: Delimiter to use
    :return: List of words
    """
    lists = [re.findall(delimiter, line) for line in open(path).readlines()]
    word_list = []
    for l in lists:
        word_list += l
    return word_list


def split_line_to_components(line):
    line = line.rstrip()
    mod = 0
    if line[-1] == '"':
        mod = 1
    reversedLine = line[-1*(2+mod)::-1]
    firstIndex = reversedLine.find('[')
    ids = reversedLine[:firstIndex][::-1]
    ids = [int(x) for x in ids.split(',')]
    reversedLine = reversedLine[firstIndex+(2+mod):]
    secondIndex = reversedLine.find(',')
    freq = int(reversedLine[:secondIndex][::-1])
    reversedLine = reversedLine[secondIndex+1:]
    key = reversedLine[::-1]
    if key.count('"') >= 2:
        key = key[1:-1]
        key = key.replace('""', '"')
    return key, freq, ids


def chunk_merger(folder, file_name, num_of_chunks):
    freqs_dict = {}
    ids_dict = {}
    p_line = None
    p_freq = 0
    p_ids = []
    fName = ""
    lineNum = 0
    path = folder + "\\" + file_name
    for i in range(1, num_of_chunks):
        print "---------File %s----------" % (str(i))
        lineNum = 0
        if exists(path+str(i)):
            with open(path + '0', 'r') as p_file, NamedTemporaryFile("wb", dir=folder, delete=False) as outfile, \
                    open(path + str(i), 'r') as s_file:
                out = csv.writer(outfile)
                fName = outfile.name
                read_p = True
                while True:  # while s_file still has lines
                    finish = False
                    s_line = s_file.readline()
                    if not s_line:
                        break
                    s_key, s_freq, s_ids = split_line_to_components(s_line)

                    if read_p:
                        p_line = p_file.readline()
                        if not p_line:
                            while s_line:
                                s_key, s_freq, s_ids = split_line_to_components(s_line)
                                out.writerow([s_key, s_freq, s_ids])
                                s_line = s_file.readline()
                            break
                        p_key, p_freq, p_ids = split_line_to_components(p_line)

                    while p_key < s_key:  #
                        out.writerow([p_key, p_freq, p_ids])
                        p_line = p_file.readline()
                        if not p_line:
                            while s_line:
                                s_key, s_freq, s_ids = split_line_to_components(s_line)
                                out.writerow([s_key, s_freq, s_ids])
                                s_line = s_file.readline()
                            finish = True
                            break
                        p_key, p_freq, p_ids = split_line_to_components(p_line)

                    if finish:
                        break

                    if p_key == s_key:
                        p_freq += s_freq
                        p_ids += s_ids
                        out.writerow([p_key, p_freq, p_ids])
                        read_p = True
                    else:
                        out.writerow([s_key, s_freq, s_ids])
                        read_p = False

                    lineNum+=1
                    if lineNum%100000==0:
                        print lineNum

                while p_line:
                    p_key, p_freq, p_ids = split_line_to_components(p_line)
                    out.writerow([p_key, p_freq, p_ids])
                    p_line = p_file.readline()
            remove(path + str(i))
            remove(path + '0')
            rename(fName, path + '0')


def merge(key=None, *iterables):
    # based on code posted by Scott David Daniels in c.l.p.
    # http://groups.google.com/group/comp.lang.python/msg/484f01f1ea3c832d

    if key is None:
        keyed_iterables = iterables
    else:
        keyed_iterables = [(Keyed(key(obj), obj) for obj in iterable)
                           for iterable in iterables]

    for element in heapq.merge(*keyed_iterables):
        yield element.obj


def batch_sort(input, output, key=None, buffer_size=128000, tempdirs=None):
    if tempdirs is None:
        tempdirs = []
    if not tempdirs:
        tempdirs.append(gettempdir())

    chunks = []
    try:
        with open(input, 'rb', 64 * 1024) as input_file:
            input_iterator = iter(input_file)
            for tempdir in cycle(tempdirs):
                current_chunk = list(islice(input_iterator, buffer_size))
                if not current_chunk:
                    break
                current_chunk.sort(key=key)
                output_chunk = open(os.path.join(tempdir, '%06i' % len(chunks)), 'w+b', 64 * 1024)
                chunks.append(output_chunk)
                output_chunk.writelines(current_chunk)
                output_chunk.flush()
                output_chunk.seek(0)
        with open(output, 'wb', 64 * 1024) as output_file:
            output_file.writelines(merge(key, *chunks))
    finally:
        for chunk in chunks:
            try:
                chunk.close()
                os.remove(chunk.name)
            except Exception:
                pass


def decrypt_id(file_id, path_to_map):
    with open(path_to_map) as f:
        for line in f:
            split_line = line.rstrip().split(",")
            if int(split_line[0]) == file_id:
                return ",".join(split_line[3:]), split_line[1], split_line[2]


def make_table(from_token_number, total_tokens_used, token_freq_path, save_path, merged_file_to_use, id_map_path, word_getter):
    from ConstantDefinitions import CLASS_MAP_PATH
    with open(CLASS_MAP_PATH, 'r') as f:
        familyToId = {line.rstrip().split(";")[1]: int(line.rstrip().split(";")[0]) for line in f.readlines()}
    varietyMap = lambda str: int(str[len("variety"):])

    with open(save_path, 'wb') as o:
        out = csv.writer(o)
        out.writerow(["Classification Matrix"])
        with open(token_freq_path) as f:
            i = 0
            tokens = []
            features = ["features"]
            tokens_to_ids = {}
            for line in f:
                if i >= from_token_number:
                    if i - from_token_number == total_tokens_used:
                        break
                    token, freq, ids = split_line_to_components(line)
                    tokens.append(token)
                    token_id = i - from_token_number
                    tokens_to_ids[token] = token_id
                    features.append("%s;%s;%s" % (token, str(token_id), str(freq)))
                i += 1
            tokens = set(tokens)
            out.writerow(features)
            features = []

        print "tokens ready"

        matrix = [0] * APK_COUNT
        for file_id in range(APK_COUNT):
            apk_path, family, variety = decrypt_id(file_id,id_map_path)
            apk_path += "\\" + merged_file_to_use
            row = []
            row.append(str(file_id))
            row.append(familyToId[family])
            row.append(varietyMap(variety))
            words = word_getter(apk_path)
            c = Counter(words)
            tokens_in_file = tokens.intersection(set(words))
            tokens_in_file = list(tokens_in_file)
            tokens_in_file = sorted(tokens_in_file, key=lambda x: tokens_to_ids[x])
            for token in tokens_in_file:
                row.append("%s;%s" % (str(tokens_to_ids[token]), c[token]))

            out.writerow(row)
            print "%s added to matrix" % str(file_id)


def make_table_from_file(file_path, sw_percent, token_count, folder):
    i = 0
    with open(file_path, 'r') as f, open(folder + "\\matrix_sw%s_token%s.csv" % (sw_percent, token_count), 'wb') as o:
        out = csv.writer(o)
        out.writerow(["Classification Matrix"])
        sw_count = 0
        for line in f:
            if i == 1:
                line = [x.split(';') for x in line.rstrip().split(',')][1:]
                counter = 0
                row = ["Features"]
                for l in line:
                    if float(l[2])/APK_COUNT > sw_percent:
                        sw_count += 1
                        continue
                    elif counter != token_count:
                        counter += 1
                        row.append("%s;%s;%s" % (l[0],l[1],l[2]))
                out.writerow(row)
            elif i > 1:
                line = line.rstrip().split(',')
                id, family, variety = line[0], line[1], line[2]
                row = [id, family, variety]
                for x in line[3:]:
                    x = x.split(';')
                    id, tf = int(x[0]), int(x[1])
                    if id < sw_count:
                        continue
                    if id >= token_count + sw_count:
                        break
                    row.append("%s;%s" % (id, tf))
                out.writerow(row)
            i += 1
    print "made table for %s-%s" % (sw_percent, token_count)


def write_matrix_to_file(matrixCopy, apk_ids, families, varieties, feature_line, out_path):
    s = ""
    print "creating %s" % out_path
    with open(out_path, 'wb') as out_file:
        out = csv.writer(out_file)
        out.writerow(["Classification Matrix"])
        out.writerow(feature_line.split(","))
        for i in range(len(apk_ids)):
            toWrite = [apk_ids[i],families[i],varieties[i]]
            row = matrixCopy[i]
            for cell in row:
                toWrite.append(cell)
            out.writerow(toWrite)


def convert_to_matrix(file_path, save_path):
    import numpy as np
    matrix = None
    i = 0
    apk_ids = []
    families = []
    varieties = []
    feature_line = ""
    id_to_doc_freq = {}
    tokenCount = 0
    id_to_index = {}
    index_to_id = {}
    from math import log
    tfidfFunc = lambda id, tf: tf * log(float(APK_COUNT) / id_to_doc_freq[id])
    with open(file_path, 'r') as source, open(save_path+"\\binary.csv", 'wb') as binOut,  open(save_path+"\\tf.csv", 'wb') as tfOut, \
        open(save_path + "\\tfNormal.csv", 'wb') as tfNormOut, open(save_path+"\\tfidf.csv", 'wb') as tfidfOut, \
        open(save_path + "\\tfidfNormal.csv", 'wb') as tfidfNormOut:
            binary = csv.writer(binOut)
            tfOut = csv.writer(tfOut)
            tfNormal = csv.writer(tfNormOut)
            tfidf = csv.writer(tfidfOut)
            tfidfNormal = csv.writer(tfidfNormOut)
            binary.writerow(["Classification Matrix"])
            tfNormal.writerow(["Classification Matrix"])
            tfOut.writerow(["Classification Matrix"])
            tfidf.writerow(["Classification Matrix"])
            tfidfNormal.writerow(["Classification Matrix"])
            for line in source:
                line = line.rstrip()
                if i==1:
                    line = [x for x in line.split(',')]
                    binary.writerow(line)
                    tfNormal.writerow(line)
                    tfOut.writerow(line)
                    tfidf.writerow(line)
                    tfidfNormal.writerow(line)
                    line = [x.split(';') for x in line[1:]]
                    tokenCount = len(line)
                    id_to_doc_freq = {int(x[1]): int(x[2]) for x in line}
                    ids = sorted(id_to_doc_freq.keys())
                    for j in range(len(line)):
                        id_to_index[ids[j]] = j
                        index_to_id[j] = ids[j]
                    id_to_doc_freq = {int(x[1]):int(x[2]) for x in line}
                elif i>1:
                    row = np.zeros(shape=tokenCount)
                    line = line.rstrip().split(',')
                    if (i-1)%1000 == 0:
                        print "Matrix conversion - line %s/%s" % ((i-1),APK_COUNT)
                    id, family, variety = line[0], line[1], line[2]

                    for x in line[3:]:
                        x = x.split(';')
                        tokenId, tf = int(x[0]), int(x[1])
                        row[id_to_index[tokenId]] = tf
                    rowCopy = np.copy(row)
                    rowCopy[rowCopy > 0] = 1
                    binary.writerow([id,family,variety] + rowCopy.tolist())
                    rowCopy = np.copy(row)
                    maxTf = rowCopy.max()
                    if maxTf != 0:
                        rowCopy /= maxTf
                    tfNormal.writerow([id,family,variety] + rowCopy.tolist())
                    tfOut.writerow([id,family,variety] + row.tolist())
                    rowCopy = np.copy(row)
                    for j in range(len(rowCopy)):
                        rowCopy[j] = tfidfFunc(index_to_id[j],rowCopy[j])
                    tfidf.writerow([id,family,variety] + rowCopy.tolist())
                    if maxTf != 0:
                        rowCopy /= maxTf
                    tfidfNormal.writerow([id,family,variety] + rowCopy.tolist())
                i+=1


def convert_to_matrix_dic(file_path, save_path):
    import numpy as np
    matrix = None
    i = 0
    apk_ids = []
    families = []
    varieties = []
    feature_line = ""
    id_to_doc_freq = {}
    id_to_index = {}
    index_to_id = {}
    with open(file_path, 'r') as source:
        for line in source:
            line = line.rstrip()
            if i==1:
                feature_line = line
                line = [x.split(';') for x in line.split(',')][1:]
                tokenCount = len(line)
                id_to_doc_freq = {int(x[1]):int(x[2]) for x in line}
                ids = sorted(id_to_doc_freq.keys())
                for j in range(len(line)):
                    id_to_index[ids[j]] = j
                    index_to_id[j] = ids[j]
                matrix = np.zeros(shape=(APK_COUNT,tokenCount))
            elif i>1:
                line = line.rstrip().split(',')
                if (i-1)%1000 == 0:
                    print "Matrix conversion - line %s/%s" % ((i-1),APK_COUNT)
                id, family, variety = line[0], line[1], line[2]
                apk_ids.append(id)
                families.append(family)
                varieties.append(variety)
                skip = False
                line = sorted(list(set(line[3:])),key= lambda x: int(x.split(";")[0]))
                for x in line:
                    x = x.split(';')
                    id, tf = int(x[0]), int(x[1])
                    matrix[i-2][id_to_index[id]] = tf
            i+=1

    matrixCopy = np.copy(matrix)
    matrixCopy[matrixCopy > 0] = 1
    write_matrix_to_file(matrixCopy,apk_ids,families,varieties,feature_line, save_path + "\\binary.csv")
    matrixCopy = np.copy(matrix)
    maxTfs = matrixCopy.max(1)
    for i in range(matrixCopy.shape[0]):
        matrixCopy[i] /= maxTfs[i]
    write_matrix_to_file(matrixCopy, apk_ids, families, varieties, feature_line, save_path + "\\tfNormal.csv")
    write_matrix_to_file(matrix, apk_ids, families, varieties, feature_line, save_path + "\\tf.csv")
    from math import log
    tfidf = lambda id,tf: tf * log(float(APK_COUNT) / id_to_doc_freq[id])
    rowNum = 0
    for row in matrix:
        cellNum = 0
        for id in row:
            matrix[rowNum][cellNum] = tfidf(index_to_id[cellNum],matrix[rowNum][cellNum])
            cellNum += 1
        rowNum += 1
    write_matrix_to_file(matrix, apk_ids, families, varieties, feature_line, save_path + "\\tfidf.csv")
    for i in range(matrix.shape[0]):
        matrix[i] /= maxTfs[i]
    write_matrix_to_file(matrix, apk_ids, families, varieties, feature_line, save_path + "\\tfidfNormal.csv")


def remove_nan_lines(save_path):
    rename(save_path + "\\tfNormal.csv", save_path + "\\tfNormalIn.csv")
    rename(save_path + "\\binary.csv", save_path + "\\binaryIn.csv")
    rename(save_path + "\\tf.csv", save_path + "\\tfIn.csv")
    rename(save_path + "\\tfidf.csv", save_path + "\\tfidfIn.csv")
    rename(save_path + "\\tfidfNormal.csv", save_path + "\\tfidfNormalIn.csv")
    with open(save_path + "\\binary.csv", 'wb') as binOut, open(save_path + "\\tf.csv",'wb') as tfOut, \
            open(save_path + "\\tfNormal.csv", 'wb') as tfNormOut, open(save_path + "\\tfidf.csv", 'wb') as tfidfOut, \
            open(save_path + "\\tfidfNormal.csv", 'wb') as tfidfNormOut, open(save_path + "\\binaryIn.csv", 'r') as binIn, \
        open(save_path + "\\tfIn.csv",'r') as tfIn, \
            open(save_path + "\\tfNormalIn.csv", 'r') as tfNormIn, open(save_path + "\\tfidfIn.csv", 'r') as tfidfIn, \
            open(save_path + "\\tfidfNormalIn.csv", 'r') as tfidfNormIn:
        i=0
        for binLine in binIn:
            tfLine = tfIn.readline()
            tfidfLine = tfidfIn.readline()
            tfNormLine = tfNormIn.readline()
            tfidfNormLine = tfidfNormIn.readline()
            if i<=1:
                binOut.write(binLine)
                tfOut.write(tfLine)
                tfNormOut.write(tfNormLine)
                tfidfOut.write(tfidfLine)
                tfidfNormOut.write(tfidfNormLine)
            else:
                val = [float(x) for x in tfNormLine.rstrip().split(',')[3:]]
                if (i - 1) % 1000 == 0:
                    print "Matrix transmutation - line %s/%s" % ((i - 1), APK_COUNT)
                if sum(val) != 0:
                    binOut.write(binLine)
                    tfOut.write(tfLine)
                    tfNormOut.write(tfNormLine)
                    tfidfOut.write(tfidfLine)
                    tfidfNormOut.write(tfidfNormLine)

            i+=1
    remove(save_path + "\\tfNormalIn.csv")
    remove(save_path + "\\binaryIn.csv")
    remove(save_path + "\\tfIn.csv")
    remove(save_path + "\\tfidfIn.csv")
    remove(save_path + "\\tfidfNormalIn.csv")


def count_empty_rows(file_path):
    empty = 0
    i = 0
    with open(file_path, 'r') as f:
        rows = [["Classification Matrix"]]
        for line in f:
            if i > 1:
                line = line.split(',')
                if len(line) == 3:
                    empty += 1
            i += 1
    print empty


def write_sample(main_file, out_path, token_count):
    percent = -1
    i=0
    with open(main_file, 'r') as f, open(out_path, 'wb') as o:
        out = csv.writer(o)
        for line in f:
            if i == token_count:
                break
            else:
                key, freq, ids = split_line_to_components(line)
                out.writerow([key, freq, float(freq) / APK_COUNT])
            i += 1
            if 100*float(i)/token_count>percent:
                percent += 1
                print str(percent) + "%"


if __name__ == "__main__":
    write_sample("D:\\Source\\Results\\tokens\\batch.csv","D:\Source\Results\\tokens\\sample.csv",5000)
    '''
    from os import makedirs
    for ngram in [2,4,5]:
        for token_count in [500,1000,1500]:
            for sw_percent in [0.85,0.9,0.95]:
                if token_count == 500 and sw_percent == 0.95:
                    continue
                for token_sw in [0.85,0.9,0.95]:
                    for token_sw_count in [500,1000,1500]:
                        path = "F:\\Results\\%sngram\\token_%s_%s\\sw_percent_%s\\matrix_sw%s_token%s" % (ngram,token_count,sw_percent,token_sw,token_sw,token_sw_count)
                        if exists(path + ".csv"):
                            new_path = path
                            if not exists(new_path):
                                makedirs(new_path)
                            convert_to_matrix_dic(path + ".csv",new_path)
    '''