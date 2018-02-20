import re
from os.path import isfile, join
from os import listdir, remove, rmdir
import time
import operator
import csv
from math import ceil
from ConstantDefinitions import CHUNK_SIZE, DELIMITER, DELIMITER_TOKEN, VISIT_PATH, ID_MAP_PATH, \
    PATH_TO_APK_ID_MAP, SPECIFIC_MERGED_JAVA, MERGED_JAVA, APK_COUNT, UTILITY_PATH, MERGED_JAVA_USABLE
from Utility import split_line_to_components, get_words_from_file, chunk_merger, batch_sort, make_table, \
    make_table_from_file

apkCount = 0
idMapString = ""

startFromCount = 0
chunkNum = startFromCount // CHUNK_SIZE

STORAGE_PATH = VISIT_PATH + "\\Results\\tokens"
ID_MAP_PATH = STORAGE_PATH + "\\idMap.txt"
tokenToFreqDictionary = {}
tokenToIdsDictionary = {}


def word_list_to_set(word_list, doc_frequency_mode=True):
    """
    Turns a list of words into a list whereupon every word may appear once.
    In docFreqMode, word appearances are reset when the file delimiter is met.
    :param word_list: Word list to work with
    :param doc_frequency_mode: True for on
    :return: List of words
    """
    words = []
    if not doc_frequency_mode:
        for doc in " ".join(word_list).split(DELIMITER_TOKEN):
            doc = list(set(doc.split( )))
            words += doc
    else:
        word_list = set(word_list)
        try:
            word_list.remove(DELIMITER_TOKEN)
        except Exception:
            pass
        words = list(word_list)
    return words


def get_unique_words_from_file(path):
    """
    A shortcut to activate both CountWords and GetWordsList
    :param path: 
    :return: 
    """
    return word_list_to_set(get_words_from_file(path))


def merge_to_frequency_dictionary(word_list):
    """
    Merges the word list with the token frequency dictionary
    :param word_list: 
    :return: 
    """
    for w in word_list:
        if w in tokenToFreqDictionary:
            tokenToFreqDictionary[w] += 1
        else:
            tokenToFreqDictionary[w] = 1


def merge_to_id_dictionary(tokens):
    """
    Adds the current file id to the end of the list of every token in the wordSet
    :param tokens: 
    :return: 
    """
    global apkCount
    for token in tokens:
        if token in tokenToIdsDictionary:
            tokenToIdsDictionary[token].append(apkCount)
        else:
            tokenToIdsDictionary[token] = [apkCount]


def add_file(path):
    """
    Defines the behaivor to act upon when a java file is found
    :param path:
    :return:
    """
    words = get_unique_words_from_file(path)
    merge_to_id_dictionary(set(words))
    merge_to_frequency_dictionary(words)


def visit_apk(path, only_check_specific, specific_folder, specific_file_name, files_merged, root=True):
    """
    For every word in the apk or in the java files in the apk folder, increase its frequency by one.
    :param path:
    :param only_check_specific: True if we want to ignore certain folders
    :param specific_folder: The specific folder to consider root, only useful when onlyCheckSpecific is on
    :param files_merged: If True, skips recursively visiting the folders in the apk and uses only the merged file.
    Note that if the variable is defined as False, then tokens are unique per document and not per apk!
    :param root: Denoted as True if this is the root folder for the apk.
    :return:
    """
    if files_merged:
        if only_check_specific:
            add_file(path + "\\" + specific_file_name)
        else:
            add_file(path + "\\" + MERGED_JAVA)
        return
    nodes = listdir(path)
    if root and only_check_specific and specific_folder in nodes:
        visit_apk(join(path, specific_folder), only_check_specific, specific_folder, specific_file_name, False, False)
    else:
        for node in nodes:
            new_path = join(path,node)
            if isfile(new_path):
                add_file(new_path)
            else:
                visit_apk(new_path, only_check_specific, specific_folder, specific_file_name, False, False)


def visit_folders(folder, result_path, id_map_path, apks_in_chunk, from_word, up_to_word, suffix=".apk",
                  only_check_specific=True, specific_folder="com", specific_file_name=MERGED_JAVA_USABLE,
                  files_merged=False, root=True):
    """
    Recursively visit a folder to find the apk folders, as denoted by a certain suffix.
    :param folder: path to the root folder
    :param from_word: value from which to start scanning
    :param up_to_word: value up to which we scan
    :param suffix: Suffix which identifies apk folders
    :param only_check_specific: Activation mode for VisitAPK
    :param specific_folder: Activation mode for VisitAPK
    :param files_merged: Activation mode for VisitAPK
    :param root: Used internally to denote root folder
    :return:
    """
    global apkCount, idMapString, tokenToFreqDictionary, tokenToIdsDictionary, startFromCount
    dirs = [join(folder, d) for d in listdir(folder) if not isfile(join(folder, d))]
    for d in dirs:
        print d
        folder_split = d.split('\\')
        if len(folder_split[-1]) >= len(suffix) and folder_split[-1][-1 * len(suffix):] == suffix and not isfile(folder):  # found apk folder
            if startFromCount != 0:
                startFromCount -= 1
            else:
                idMapString += str(apkCount) + "," + folder_split[-3] + "," + folder_split[-2] + "," + d + "\r\n"
                visit_apk(d, only_check_specific, specific_folder, specific_file_name, files_merged)
                apkCount = apkCount + 1
                if apkCount % apks_in_chunk == 0:
                    save_current_results(result_path)
        elif (up_to_word > folder_split[-1] >= from_word) or \
                folder_split[-1][:-1] == "variety" or folder_split[-1][:-2] == "variety":
            visit_folders(d, result_path, id_map_path, apks_in_chunk, from_word, up_to_word, suffix,
                          only_check_specific, specific_folder, specific_file_name, files_merged, False)

    if root:
        save_current_results(result_path)
        with open(id_map_path, 'w') as f:
            f.write(idMapString)


def save_current_results(result_path):
    global tokenToFreqDictionary, tokenToIdsDictionary, chunkNum
    sorted_dict = sorted(tokenToFreqDictionary.items(), key=operator.itemgetter(0))
    lst = [(i[0], str(i[1]), tokenToIdsDictionary[i[0]]) for i in sorted_dict if i[0] != DELIMITER_TOKEN]

    o = open(result_path + str(chunkNum), 'wb')
    out = csv.writer(o)
    out.writerows(lst)
    tokenToFreqDictionary = {}
    tokenToIdsDictionary = {}
    chunkNum += 1


def count_apks(folder, id_map_path, from_word, up_to_word, suffix=".apk", root=True):
    """
    Recursively visit a folder to find the apk folders, as denoted by a certain suffix.
    :param folder: path to the root folder
    :param from_word: value from which to start scanning
    :param up_to_word: value up to which we scan
    :param suffix: Suffix which identifies apk folders
    :param root: Used internally to denote root folder
    :return:
    """
    global apkCount, idMapString, tokenToFreqDictionary, tokenToIdsDictionary, startFromCount
    dirs = [join(folder, d) for d in listdir(folder) if not isfile(join(folder, d))]
    for d in dirs:
        folder_split = d.split('\\')
        if (up_to_word > folder_split[-1] >= from_word) or \
                        folder_split[-1][:-1] == "variety" or folder_split[-1][:-2] == "variety":
            count_apks(d, id_map_path,from_word, up_to_word, suffix,False)
        elif folder_split[-1][-1 * len(suffix):] == suffix and not isfile(folder):  # found apk folder
            if startFromCount != 0:
                startFromCount -= 1
            else:
                idMapString += str(apkCount) + "," + folder_split[-3] + "," + folder_split[-2] + "," + d + "\r\n"
                remove(d + "\\" + SPECIFIC_MERGED_JAVA)
                apkCount = apkCount + 1
    if root:
        with open(id_map_path, 'w') as f:
            f.write(idMapString)


def merge_apk(path, only_check_specific, specific_folder, out, delete_file_mode, root=True):
    """
    Merges all java files in the apk folder into one java file
    :param path: The current folder path
    :param only_check_specific: See VisitAPK
    :param specific_folder: See VisitAPK
    :param out: The file into which all documents are merged
    :param delete_file_mode: Whether to delete files after they are merged
    :param root:
    :return:
    """
    nodes = listdir(path)
    try:
        for node in nodes:
            new_path = join(path, node)
            if new_path.split('\\')[-1] == MERGED_JAVA or new_path.split('\\')[-1] == SPECIFIC_MERGED_JAVA:
                continue
            if isfile(new_path):
                if new_path.split("\\")[-1].split(".")[-1] == "java":
                    in_file = open(new_path, 'r')
                    out.write(new_path + "\r\n")
                    out.write(in_file.read() + DELIMITER)
                    in_file.close()
                if delete_file_mode:
                    remove(new_path)
            else:
                merge_apk(new_path, only_check_specific, specific_folder, out, delete_file_mode, False)
                if delete_file_mode:
                    rmdir(new_path)
    except Exception:
        pass


def create_specific_merged_file(folder, out_name, specific_folder):
    merged_file = open(folder + "\\" + MERGED_JAVA, 'r')
    out = open(folder + "\\" + out_name, 'w')
    merged_text = merged_file.read().split("\r\n")
    first = True
    ignore = True
    wrote = False
    txt = []
    for line in merged_text:
        if line == DELIMITER_TOKEN:
            ignore = True
            first = True
            if wrote:
                txt.append(DELIMITER_TOKEN)
            wrote = False
        else:
            line_split = line.split("\\")
            if first and specific_folder in line_split:
                ignore = False
            elif not ignore and not first:
                wrote = True
                txt.append(line)
            first = False
    txt = "\r\n".join(txt)
    out.write(txt)
    out.close()
    merged_file.close()


def merge_apks_in_folders(folder, from_word, up_to_word, suffix=".apk", only_check_specific=True, specific_folder="com",
                          delete=False):
    """Recursively finds apk folder in which documents are merged"""

    dirs = [join(folder, d) for d in listdir(folder) if not isfile(join(folder, d))]
    for d in dirs:
        print d
        folder_split = d.split('\\')
        if (up_to_word > folder_split[-1] >= from_word) or \
                        folder_split[-1][:-1] == "variety" or folder_split[-1][:-2] == "variety":
            merge_apks_in_folders(d, from_word, up_to_word, suffix, only_check_specific, specific_folder, delete)
        else:
            if folder_split[-1][-1*len(suffix):] == suffix and not isfile(folder):  # found apk folder
                nodes = listdir(d)
                flag = 0
                if len(nodes) == 2:
                    for node in nodes:
                        if node == MERGED_JAVA:
                            flag += 1
                        elif node == SPECIFIC_MERGED_JAVA:
                            flag += 1
                if flag != 2:
                    out = open(d + "\\" + MERGED_JAVA, 'w')
                    merge_apk(d, only_check_specific, specific_folder, out, delete)
                    out.close()
                    if only_check_specific:
                        create_specific_merged_file(d, SPECIFIC_MERGED_JAVA, specific_folder)


def delete_paths_from_merged_file(folder, from_word, up_to_word, suffix=".apk", out_name=MERGED_JAVA_USABLE):
    """Recursively finds apk folder in which documents are merged"""
    dirs = [join(folder, d) for d in listdir(folder) if not isfile(join(folder, d))]
    for d in dirs:
        print d
        folder_split = d.split('\\')
        if (up_to_word > folder_split[-1] >= from_word) or \
                        folder_split[-1][:-1] == "variety" or folder_split[-1][:-2] == "variety":
            delete_paths_from_merged_file(d, from_word, up_to_word, suffix)
        else:
            if folder_split[-1][-1*len(suffix):] == suffix and not isfile(folder):  # found apk folder
                create_specific_merged_file(d, out_name, folder_split[-1])   # every apk is used this way

def getWeirdFiles(path):
    s = set()
    j=0
    for i in range(APK_COUNT):
        s.add(i)
    with open(path) as f:
        for line in f:
            j+=1
            key, freq, ids = split_line_to_components(line)
            idSet = set(ids)
            oddOnes = s.difference(idSet)
            print oddOnes
            print key
            if j==5:
                break


if __name__ == "__main__":
    start = time.time()


    #save_path = "F:\\tokenExperiment"  #  change this value to the folder you place the file matrixWhole.csv at.
    save_path = "D:\\AMData\\ComplexLab\\2gram\\0.8\\ST"


    '''
    #visit_folders(VISIT_PATH, STORAGE_PATH + "\\resultMap.part", ID_MAP_PATH, CHUNK_SIZE, "A", "Zz", only_check_specific=True,
                  #files_merged=True, specific_file_name=MERGED_JAVA_USABLE)
    #chunk_merger(STORAGE_PATH, "resultMap.part", int(ceil(float(APK_COUNT) / CHUNK_SIZE)))
    #batch_sort(STORAGE_PATH + "\\resultMap.part0", STORAGE_PATH + "\\batch.csv",
              # key=lambda x: -1 * int(split_line_to_components(x)[1]))
    #save_path = input_path
    #id_map_path = save_path + "\\idMap.txt"
    #sorted_tokens = save_path + "\\batch.csv"
    #make_table(0, 100000, sorted_tokens, save_path + "\\matrixWhole.csv", MERGED_JAVA_USABLE,
    #           id_map_path, get_words_from_file)
    '''
    from os.path import exists
    from os import makedirs
    for sw_percent in [0.4, 0.6, 0.8, 1]:#0.4
        for token_count in [10000]:
            path = "%s\\%s_%s" % (save_path, sw_percent, token_count)
            if not exists(path):
                makedirs(path)
            print "%s_%s" % (sw_percent, token_count)
            make_table_from_file(save_path + "\\matrixWhole.csv", sw_percent, token_count, path)
            from Utility import convert_to_matrix, remove_nan_lines
            #remove_nan_lines(save_path)
            convert_to_matrix(path+"\\matrix_sw%s_token%s.csv" % (sw_percent,token_count),path)

    time_str = "--- run time: %s seconds ---\r\n" % (time.time() - start)
    count_str = "--- apk count: %s apks ---\r\n" % apkCount
    perf = open(UTILITY_PATH, 'w')
    perf.write(time_str)
    perf.write(count_str)





