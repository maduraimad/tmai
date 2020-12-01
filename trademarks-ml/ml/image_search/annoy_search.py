from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np

# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}

dims = 2048
n_nearest_neighbors = 1001
trees = 2
infiles = glob.glob('/data2/trademarksImageSearch/indexer/model_index/*.npz')
# build ann index
t = AnnoyIndex(dims)
for file_index, i in enumerate(infiles):
  #print (file_index)
  #print (i)
  file_vector = np.loadtxt(i,delimiter=',')
  file_name = os.path.basename(i).split('.')[0]
  file_index_to_file_name[file_index] = file_name
  file_index_to_file_vector[file_index] = file_vector
  t.add_item(file_index, file_vector)
t.build(trees)
t.save('model_index.ann')

# create a nearest neighbors json file for each input
if not os.path.exists('nearest_neighbors_model'):
  os.makedirs('nearest_neighbors_model')

for i in file_index_to_file_name.keys():
  master_file_name = file_index_to_file_name[i]
  master_vector = file_index_to_file_vector[i]

  named_nearest_neighbors = []
  nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
  for j in nearest_neighbors:
    neighbor_file_name = file_index_to_file_name[j]
    neighbor_file_vector = file_index_to_file_vector[j]

    similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
    rounded_similarity = int((similarity * 10000)) / 10000.0

    named_nearest_neighbors.append({
      'filename': neighbor_file_name,
      'similarity': rounded_similarity
    })

  with open('nearest_neighbors_model/' + master_file_name + '.json', 'w') as out:
    json.dump(named_nearest_neighbors, out)
