import glob
import json
import pickle
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering
import igraph

json_folder = "/Users/greensod/usptoWork/TrademarkRefiles/data/tsdrJsons/**/*.json"
pickle_file = "/Users/greensod/usptoWork/TrademarkRefiles/data/tsdrJsons/output.pickle"
cluster_output_file = "/Users/greensod/usptoWork/TrademarkRefiles/data/tsdrJsons/cluster_output.json"
# json_folder = "/home/ubuntu/data/tsdrjsondata/**/*.json"
# pickle_file = "/home/ubuntu/data/tsdrjsondata/designCodeToTrademarksMapping.pickle"
# cluster_output_file = "/home/ubuntu/data/tsdrjsondata/clusterOutput_100.json"
reduced_dimensions = 1000
design_code_to_patent_map = {}
designCodes = []
patentsList = []
encodedPatentsList = None
design_code_map = None # this will be a np array

def createClustersUsingGraphMethod(numOfClusters):
    createDesginCodeToDesignCodeMapping()
    shape = np.shape(design_code_map)
    g = igraph.Graph()
    for row in range(shape[0]):
        g.add_vertex(name=designCodes[row])
    for row in range(shape[0]):
        print("Row - "+str(row))
        for column in range(row,shape[1]):
            if(row != column):
                g.add_edge(row, column, weight= design_code_map[row, column])
    cluster = g.community_fastgreedy(weights="weight")
    subgraphs = cluster.as_clustering(numOfClusters).subgraphs()
    print("###Printing clusters###")
    for subgraph in subgraphs:
        print(subgraph.vs["name"])


def createDesginCodeToDesignCodeMapping():
    print("Creating design code to design code mapping")
    global design_code_map
    global designCodes
    readDesignCodeMap()
    designCodes = list(design_code_to_patent_map.keys())
    totalDesignCodes = len(designCodes)
    arr = np.zeros((totalDesignCodes, totalDesignCodes))
    for index1, design_code_1 in enumerate(designCodes):
        non_zero_counts = []
        for index2, design_code_2 in enumerate(designCodes):
            if(design_code_1 != design_code_2):
                serial_numbers_1 = design_code_to_patent_map[design_code_1];
                serial_numbers_2 = design_code_to_patent_map[design_code_2];
                common_count =  len(serial_numbers_1.intersection(serial_numbers_2))
                if(common_count > 0):
                    arr[index1, index2] = common_count
                    non_zero_counts.append(common_count)
        if(len(non_zero_counts) > 0):
            non_zero_counts_avg = sum(non_zero_counts)/len(non_zero_counts)
            # arr[index1, index1] = non_zero_counts_avg
    print("Done design code mapping")
    design_code_map = arr


def createDesignCodeToTrademarksMapping():
    json_files = glob.glob(json_folder, recursive=True)
    print(str(len(json_files)))
    count = 0
    for filename in json_files:
        count += 1
        print(str(count))
        with open(filename, 'r') as f:
            json_data = json.load(f)
            trademark = json_data["trademarks"][0]
            serial_number = trademark["status"]["serialNumber"]
            designSearchList = trademark["status"]["designSearchList"]
            designCodes = [item["code"] for item in designSearchList]
            for code in designCodes:
                if(not code in design_code_to_patent_map):
                    design_code_to_patent_map[code] = set()
                design_code_to_patent_map[code].add(serial_number)
    print("Writing to file")
    with open(pickle_file, 'wb') as file:
        pickle.dump(design_code_to_patent_map, file)

def printStats():
    readDesignCodeMap()
    print("Total design codes - " + str(len(design_code_to_patent_map.keys())))

def createClusterElbowPlot():
    createDesginCodeToDesignCodeMapping()
    clusterSizeArray = []
    inertiaArray = []
    for clusterSize in range(50,200, 10):
        print("size - "+str(clusterSize))
        clusterSizeArray.append(clusterSize)
        kmeans = KMeans(n_clusters=clusterSize, random_state=0, verbose=1, precompute_distances=True, n_jobs=-1).fit(design_code_map)
        inertiaArray.append(kmeans.inertia_)
    plt.plot(clusterSizeArray, inertiaArray)
    plt.show()

def createEncodedPatentList():
    print("Loading pickle mapping file")
    global encodedPatentsList
    readDesignCodeMap()
    for designCode in design_code_to_patent_map:
        designCodes.append(designCode)
        patentsList.append(design_code_to_patent_map[designCode])
    lb = MultiLabelBinarizer()
    lb = lb.fit(patentsList)
    encodedPatentsList = lb.transform(patentsList)
    print("Finished loading pickle mapping file ")


def createClusters(clusterSize):
    createDesginCodeToDesignCodeMapping()
    # reducedEncodedPatentsList = reduceDimensions()
    print("Starting kmeans clustering")
    # kmeans = KMeans(n_clusters=clusterSize, random_state=0, verbose=1, precompute_distances=True, n_jobs=-1).fit(design_code_map)
    # kmeans = KMeans(n_clusters=clusterSize, random_state=0, verbose=1, n_init=1).fit(design_code_map)
    kmeans = SpectralClustering(clusterSize, affinity='precomputed', n_init=100)
    kmeans.fit(design_code_map)
    print("Finished cluster")
    kmeansLabels = kmeans.labels_
    kmeansOutput = {}
    i = 0
    for label in kmeansLabels:
        if (not str(label) in kmeansOutput):
            kmeansOutput[str(label)] = []
        kmeansOutput[str(label)].append(designCodes[i])
        i += 1
    print(kmeansOutput)
    with open(cluster_output_file, 'w') as file:
        file.write(json.dumps(kmeansOutput))

def reduceDimensions():
    print("Reducing dimensions")
    svd = TruncatedSVD(n_components=reduced_dimensions, n_iter=7, random_state=42)
    svd.fit(encodedPatentsList)
    new_x = svd.transform(encodedPatentsList)
    print("Finished reducing dimensions")
    return new_x


def readDesignCodeMap():
    with open(pickle_file, "rb") as file:
        global design_code_to_patent_map
        design_code_to_patent_map  = pickle.load(file)

def samplePlot():
    x = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400]
    y = [95719108.85527034,76410892.58141798,61472079.04806905,52641741.408994645,45566092.931076445,40227246.44645356,35436067.741320364,31438500.498737786,27751721.60969257,24020407.888051026,21295988.12082343,18920117.399244037,17120121.73152519,15601388.804601204,14251669.838095253,13237384.749613002,12081144.081263263,11040371.580638759,10085788.14293178,9343426.54943418,8593488.455111824,7927689.069917039,7405667.990778306,6920791.909909548,6425526.576833142,5982495.638461074,5500868.320006338,5200051.550054918,4868071.544784074,4559014.9970196,4291501.105924338,4025268.147414206,3785977.2583221854,3564146.0326404907,3361051.431174505,3165108.283828672]
    plt.plot(x, y)
    plt.show()



# total design codes 1377
# createDesignCodeToTrademarksMapping()
# printStats()
# createClusterElbowPlot()
# reduceDimensions()
# createClusters(100)
# createDesginCodeToDesignCodeMapping()
# samplePlot()
createClustersUsingGraphMethod(5)


