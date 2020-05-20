from scipy.spatial.distance import cosine, minkowski
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def nearest_neighbour_glove(vec, words_exception=[], number_of_neighbour=5, glove_dict = glove_twitter_dict, method='cosine'):
    distances = []
    if method=='cosine':
        distance_func = cosine
    elif method == 'mse':
        distance_func = lambda x,y: minkowski(x,y,2)
    elif method == 'mae':
        distance_func = lambda x,y: minkowski(x,y,1)
    for word in tqdm(glove_dict):
        distances.append((distance_func(vec, glove_dict[word]),word))
    distances = sorted(distances, key = lambda v: v[0])
    if distances[0][0]==0:
        distances = distances[1:]
    return distances[:number_of_neighbour]