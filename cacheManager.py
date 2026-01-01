import shutil
import os

cache_path = "tunning_cache"
layer_path = "/Layers"
shutil.rmtree(cache_path)

os.makedirs(cache_path)
os.makedirs(cache_path + layer_path)
