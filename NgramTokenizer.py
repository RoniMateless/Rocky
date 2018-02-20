from Tokenizer import get_words_from_file
from os.path import join, isfile, exists
from os import listdir, makedirs, remove
import csv
import operator

SW_TOKEN = "TOKEN_SW"
LT_TOKEN = "TOKEN_LT"
WC_TOKEN = "TOKEN_WC"

SW_FILENAME = "stopwords.csv"
LT_FILENAME = "longtail.csv"
TOKEN_FILENAME = "tokens.csv"

tokenToFreqDictionary = {}
tokenToIdsDictionary = {}

idMap = None

from ConstantDefinitions import SPECIFIC_MERGED_JAVA, MERGED_JAVA, VISIT_PATH, APK_COUNT, CHUNK_SIZE, \
    DELIMITER_TOKEN, DELIMITER, MERGED_JAVA_USABLE
from Utility import split_line_to_components, make_table, make_table_from_file, chunk_merger, batch_sort, write_sample
from math import ceil

STORAGE_PATH = VISIT_PATH + "\\Results\\ngrams"
MODE_NONE = "none"
MODE_ACE = "ace"
MODE_WILDCARD = "wild"
sw_set = set()
token_set = set()
chunkNum = 0
startFromCount = 0
idMapString = ""
apkCount = 0

def get_token_ngrams(words, size=1, mode="none", repeating = False):
    if mode == MODE_NONE:
        ngrams = []
        i = 0
        while i < len(words) - size:
            is_valid = True
            ngram = []
            for j in range(0, size):
                if words[i+j] not in token_set:
                    i += j
                    is_valid = False
                    break
                ngram.append(words[i + j])
            if is_valid:
                ngrams.append(" ".join(ngram))
            i += 1
        if repeating:
            return ngrams
        else:
            return list(set(ngrams))
    elif mode == MODE_ACE:
        def selector(x):
            if x in sw_set:
                return SW_TOKEN
            elif x in token_set:
                return x
            return LT_TOKEN
        words = [selector(x) for x in words]
    elif mode == MODE_WILDCARD:
        def selector(x):
            if x not in token_set:
                return WC_TOKEN
            return x
        words = [selector(x) for x in words]
    ngrams = []
    for i in range(0, len(words) - size):
        ngram = words[i]
        for j in range(1, size):
            ngram += " " + words[i + j]
        ngrams.append(ngram)
    if repeating:
        return ngrams
    else:
        return list(set(ngrams))


def extract_token_maps(path_to_token_frequency, sw_percent, token_count):
    i = 0
    with open(path_to_token_frequency) as f:
        for line in f:
            token, freq, ids = split_line_to_components(line)
            percent = float(freq)/APK_COUNT
            if percent >= sw_percent:
                sw_set.add(token)
            elif i <= token_count:
                i += 1
                token_set.add(token)
            else:
                break



def visit_folders(folder, result_path, id_map_path, apks_in_chunk, from_word, up_to_word, ngram_size, activation_mode,
                  suffix=".apk", only_check_specific=True, specific_folder="com", specific_file=SPECIFIC_MERGED_JAVA, files_merged=True, root=True):
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
        print "%s/%s -- %s" % (apkCount, APK_COUNT, d)
        folder_split = d.split('\\')
        if len(folder_split[-1]) >= len(suffix) and folder_split[-1][-1 * len(suffix):] == suffix and not isfile(folder):  # found apk folder
            if startFromCount != 0:
                startFromCount -= 1
            else:
                idMapString = str(apkCount) + "," + folder_split[-3] + "," + folder_split[-2] + "," + d + "\r\n"
                idMap.write(idMapString)
                visit_apk(d, only_check_specific, specific_folder, specific_file, ngram_size, activation_mode, files_merged)
                apkCount = apkCount + 1
                if apkCount % apks_in_chunk == 0:
                    save_current_results(result_path)
        elif (up_to_word > folder_split[-1] >= from_word) or \
                folder_split[-1][:-1] == "variety" or folder_split[-1][:-2] == "variety":
            visit_folders(d, result_path, id_map_path, apks_in_chunk, from_word, up_to_word, ngram_size, activation_mode, suffix,
                          only_check_specific, specific_folder, specific_file, files_merged, False)

    if root:
        save_current_results(result_path)


def visit_apk(path, only_check_specific, specific_folder, specific_file, ngram_size, activation_mode, files_merged, root=True):
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
            add_file(path + "\\" + specific_file, ngram_size, activation_mode)
        else:
            add_file(path + "\\" + MERGED_JAVA, ngram_size, activation_mode)
        return
    nodes = listdir(path)
    if root and only_check_specific and specific_folder in nodes:
        visit_apk(join(path, specific_folder), only_check_specific, specific_folder, specific_file, ngram_size,
                  activation_mode, False, False)
    else:
        for node in nodes:
            new_path = join(path,node)
            if isfile(new_path):
                add_file(new_path)
            else:
                visit_apk(new_path, only_check_specific, specific_folder, specific_file, ngram_size, activation_mode, False, False)

def save_current_results(result_path):
    global tokenToFreqDictionary, tokenToIdsDictionary, chunkNum
    sorted_dict = sorted(tokenToFreqDictionary.items(), key=operator.itemgetter(0))
    lst = [(i[0], str(i[1]), tokenToIdsDictionary[i[0]]) for i in sorted_dict if i[0] != DELIMITER_TOKEN]

    o = open(result_path + "" + str(chunkNum), 'wb')
    out = csv.writer(o)
    out.writerows(lst)
    tokenToFreqDictionary = {}
    tokenToIdsDictionary = {}
    chunkNum += 1


def add_file(path, ngram_size, activation_mode):
    """
    Defines the behaivor to act upon when a java file is found
    :param path:
    :return:
    """
    words = get_words_from_file(path)
    try:
        words = words.remove(DELIMITER_TOKEN)
    except Exception:
        pass
    words = get_token_ngrams(words, ngram_size, activation_mode)
    merge_to_id_dictionary(set(words))
    merge_to_frequency_dictionary(words)


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


def ngram_getter(ngram_size, activation_mode):
    def func(path):
        words = get_words_from_file(path)
        try:
            words = words.remove(DELIMITER_TOKEN)
        except Exception:
            pass
        return get_token_ngrams(words, ngram_size, activation_mode, repeating=True)
    return func


def run_everything(ngram_size, token_stopword_percentage, token_count, ngram_stopword_percentages, ngram_token_counts, save_path):
    global chunkNum, apkCount, idMap, startFromCount
    chunkNum = 0
    startFromCount = 0
    apkCount = 0
    save_path = save_path + "\\token_%s_%s" % (token_count, token_stopword_percentage)
    if not exists(save_path):
        makedirs(save_path)
    extract_token_maps("D:\\Source\\Results\\tokens\\batch.csv", token_stopword_percentage,token_count)
    all_tokens_filename = "ngramResultMap.part"
    all_tokens_path = save_path + "\\" + all_tokens_filename
    id_map_path = save_path + "\\ngramIdMap2.txt"
    sorted_tokens = save_path + "\\sortedTokens.csv"
    with open(id_map_path, 'w') as idMap:
        visit_folders(VISIT_PATH, all_tokens_path, id_map_path, CHUNK_SIZE, "A", "Zz", ngram_size,
                      MODE_ACE, only_check_specific=True, specific_file=MERGED_JAVA_USABLE)
    chunk_merger(save_path, all_tokens_filename, int(ceil(float(APK_COUNT) / CHUNK_SIZE)))
    batch_sort(save_path + "\\" + all_tokens_filename + "0", sorted_tokens,
               key=lambda x: -1 * int(split_line_to_components(x)[1]))
    remove(save_path + "\\" + all_tokens_filename + "0")
    write_sample(sorted_tokens, save_path + "\\ngramSample.csv", 30000)
    get_ngrams_from_file = ngram_getter(ngram_size, MODE_ACE)
    make_table(0, 50000, sorted_tokens, save_path + "\\matrixWhole.csv", MERGED_JAVA_USABLE,
               id_map_path, get_ngrams_from_file)
    for sw_percent in ngram_stopword_percentages:
        path = "%s\\sw_percent_%s" % (save_path, sw_percent)
        if not exists(path):
            makedirs(path)
        for token_count in ngram_token_counts:
            print "%s_%s" % (sw_percent, token_count)
            make_table_from_file(save_path + "\\matrixWhole.csv", sw_percent, token_count, path)


def initialize_dataset(save_path, ngram_size):
    global chunkNum, apkCount, idMap, startFromCount
    apkCount = 0

    startFromCount = apkCount
    chunkNum = apkCount / CHUNK_SIZE

    save_path = save_path + "\\%sngrams" % ngram_size
    if not exists(save_path):
        makedirs(save_path)
    extract_token_maps("D:\\Source\\Results\\tokens\\batch.csv", 1, 20000)
    print len(sw_set)
    print len(token_set)
    all_tokens_filename = "ngramResultMap.part"
    all_tokens_path = save_path + "\\" + all_tokens_filename
    id_map_path = save_path + "\\ngramIdMap.txt"
    sorted_tokens = save_path + "\\sortedTokens.csv"
    with open(id_map_path, 'w') as idMap:
        visit_folders(VISIT_PATH, all_tokens_path, id_map_path, CHUNK_SIZE, "A", "Zz", ngram_size,
                      MODE_ACE, only_check_specific=True, specific_file=MERGED_JAVA_USABLE)
    chunk_merger(save_path, all_tokens_filename, int(ceil(float(APK_COUNT) / CHUNK_SIZE)))
    batch_sort(save_path + "\\" + all_tokens_filename + "0", sorted_tokens,
               key=lambda x: -1 * int(split_line_to_components(x)[1]))
    remove(save_path + "\\" + all_tokens_filename + "0")
    get_ngrams_from_file = ngram_getter(ngram_size, MODE_ACE)
    make_table(0, 20000, sorted_tokens, save_path + "\\matrixWhole.csv", MERGED_JAVA_USABLE,
               id_map_path, get_ngrams_from_file)


def convert_to_matrix_forms(save_folder, ngram_stopword_percentages, ngram_token_counts):
    for sw_percent in ngram_stopword_percentages:
        for token_count in ngram_token_counts:
            path = "%s\\matrix_%s_%s" % (save_folder,sw_percent,token_count)
            file = path + ".csv"
            if not exists(path):
                makedirs(path)
            from Utility import convert_to_matrix
            convert_to_matrix(file,path)


def make_table_from_dataset(dataset_path, token_stopword_percentages, token_counts, ngram_stopword_percentages, ngram_token_counts, save_folder):
    tempPath = save_folder
    from ConstantDefinitions import CLASS_MAP_PATH
    with open(CLASS_MAP_PATH, 'r') as f:
        familyToId = {line.rstrip().split(";")[1]: int(line.rstrip().split(";")[0]) for line in f.readlines()}
    varietyMap = lambda str: int(str[len("variety"):])
    for token_stopword_percentage in token_stopword_percentages:
        for token_count in token_counts:
            save_folder = tempPath
            save_folder += "\\%s_%s" % (token_stopword_percentage, token_count)
            if not exists(save_folder):
                makedirs(save_folder)
            extract_token_maps("D:\\Source\\Results\\tokens\\batch.csv", token_stopword_percentage, token_count)
            ngram_token_counts = sorted(ngram_token_counts, reverse=True)
            max_token_count = ngram_token_counts[0]

            def selector(x):
                if x in sw_set:
                    return SW_TOKEN
                elif x in token_set:
                    return x
                return LT_TOKEN

            with open(dataset_path, 'r') as f:
                sw_count = 0
                i = 0
                ids_to_tokens = {}
                token_to_df = {}
                for line in f:
                    if i == 1:
                        tokens_and_ids = [x.split(';') for x in line.rstrip().split(',')][1:]
                        for x in tokens_and_ids:  # map token ids to their tokens.
                            token = " ".join([selector(word) for word in x[0].split( )])
                            ids_to_tokens[int(x[1])] = token
                    elif i > 1:
                        line = [x.split(';') for x in line.rstrip().split(',')[3:]]
                        # suspected line for memory problems
                        tokens_in_apk = set([ids_to_tokens[int(x[0])] for x in line])  # isolate unique tokens under new generic constraints
                        # end suspected line
                        print "Extracting tokens... %s" % ((i-1)*100.0/APK_COUNT) + "%"
                        for token in tokens_in_apk:  # count df for every token
                            if token in token_to_df:
                                token_to_df[token] += 1
                            else:
                                token_to_df[token] = 1
                    i += 1

            tokens_by_freq = sorted(token_to_df, key = lambda x: x[1], reverse=True)
            tokens_by_freq = [(x,token_to_df[x]) for x in tokens_by_freq]
            tokens_by_freq = sorted(tokens_by_freq, key = lambda x: x[1], reverse=True)
            print len(tokens_by_freq)
            token_to_df = {}
            tokens = []
            tokens_to_selected_ids = {}
            token_freq = []
            i = 0
            for sw_percent in ngram_stopword_percentages:
                with open(dataset_path, 'r') as f:
                    print "Creating classification matrix for stopword percent %s" % (sw_percent)
                    with open("%s\\%s_%s_%s.csv" % (save_folder,"matrix",sw_percent,max_token_count),'wb') as o:
                        out = csv.writer(o)
                        i=0
                        while True:  # Choose tokens according to specifications
                            if tokens_by_freq[i][1] < APK_COUNT * sw_percent:
                                current_tokens = tokens_by_freq[i:i + max_token_count]
                                print len(current_tokens)
                                print str(i)
                                tokens = [x[0] for x in current_tokens]
                                token_freq = [x[1] for x in current_tokens]
                                break
                            i += 1
                        out.writerow(["Classification Matrix"])
                        row = ["Features"]
                        i=0
                        for i in range(max_token_count):  # Write features row to out
                            token, freq = tokens[i], token_freq[i]
                            row.append("%s;%s;%s" % (token, int(i), freq))
                            tokens_to_selected_ids[token] = i
                        out.writerow(row)

                        i = 0
                        for line in f:
                            if i > 1:
                                line = line.rstrip().split(',')
                                id, family, variety = line[0], familyToId[line[1]], varietyMap(line[2])
                                row = [id, family, variety]
                                freqs = [0] * max_token_count
                                print "rewriting line %s/%s" % ((i-1),APK_COUNT)
                                for x in line[3:]:
                                    x = x.split(';')
                                    id, tf = int(x[0]), int(x[1])
                                    if ids_to_tokens[id] in tokens:
                                        token = ids_to_tokens[id]
                                        freqs[tokens_to_selected_ids[token]] += tf
                                for j in range(max_token_count):
                                    if freqs[j]!=0:
                                        row.append("%s;%s" % (j, freqs[j]))
                                out.writerow(row)
                            i += 1

                    for token_count in ngram_token_counts[1:]: # use previous max token count to create next file
                        with open("%s\\%s_%s_%s.csv" % (save_folder, "matrix", sw_percent, max_token_count), 'r') as f,\
                             open("%s\\%s_%s_%s.csv" % (save_folder, "matrix", sw_percent, token_count), 'wb') as o:
                            out = csv.writer(o)
                            out.writerow(["Classification Matrix"])
                            sw_count = 0
                            i = 0
                            for line in f:
                                if i == 1:
                                    line = line.rstrip().split(',')[:token_count+1]
                                    out.writerow(line)
                                elif i > 1:
                                    line = line.rstrip().split(',')[:3+token_count]
                                    for index in reversed(range(len(line))):
                                        if int(line[index].split(';')[0]) < token_count:
                                            line = line[:index+1]
                                            break
                                    out.writerow(line)
                                i += 1

            convert_to_matrix_forms(save_folder,ngram_stopword_percentages,ngram_token_counts)
            print "made table for %s-%s" % (ngram_stopword_percentages, ngram_token_counts)


if __name__ == "__main__":
    make_table_from_dataset(STORAGE_PATH + "\\datasets\\3ngrams\\matrixWhole.csv",[0.85,0.9,0.95],[500,1000,1500],[0.85,0.9,0.95],[500,1000,1500]
                           ,STORAGE_PATH + "\\datasets\\3ngrams\\matrices")
    '''
    for ngram_size in range(3, 6):
        path = "%s\\%s" % (STORAGE_PATH, "datasets")
        print path
        if not exists(path):
            makedirs(path)
        initialize_dataset(path,ngram_size)

    for ngram_size in range(2,6):
        for token_count in [500,1000,1500]:
            for stopword_percent in [0.85,0.9,0.95]:
                if token_count==500 and stopword_percent==0.95 or token_count==500 and stopword_percent==0.9:
                    continue
                path = "%s\\%sngram" % (STORAGE_PATH, ngram_size)
                if not exists(path):
                    makedirs(path)

                run_everything(ngram_size, stopword_percent, token_count, [0.85,0.9,0.95], [500,1000,1500], path)
    '''
