from fiftyone import zoo

zoo.download_zoo_dataset('coco-2017', splits = ['train', 'validation'], shuffle = True, seed = 1234, dataset_dir = './data/')