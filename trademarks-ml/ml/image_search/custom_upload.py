from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random,pickle
import numpy as np

# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}

infiles = glob.glob('/data2/trademarksImageSearch/indexer/model_index/*.npz')
for file_index, i in enumerate(infiles):
  file_vector = np.loadtxt(i,delimiter=',')
  file_name = os.path.basename(i).split('.')[0]
  file_index_to_file_name[file_index] = file_name
  file_index_to_file_vector[file_index] = file_vector

pickle.dump(file_index_to_file_name, open("file_index_to_file_name.pickle", "wb"))
pickle.dump(file_index_to_file_vector, open("file_index_to_file_vector.pickle", "wb"))
