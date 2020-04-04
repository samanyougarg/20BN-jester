import pandas as pd
import numpy as np

class DataLoader():
    """ Class used to load csvs
    # Arguments
        path_vid    : path to the folder containing the extracted Jester dataset
        path_labels : path to the csv containing the labels
        path_train  : path to the csv containing the list of videos for training
        path_val    : path to the csv containing the list of videos used for validation
        path_test   : path to the csv containing the list of videos used for test
    # Returns
        An instance of the DataLoader class  
    """
    def __init__(self, path_vid, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_vid    = path_vid
        self.path_labels = path_labels
        self.path_train  = path_train
        self.path_val    = path_val
        self.path_test   = path_test

        self.get_labels(path_label s)

        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)

        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)

        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")

    def get_labels(self, path_labels):
        """Loads the labels dataframe from a csv and creates dictionaries to convert the string labels to int and backwards
        # Arguments
            path_labels : path to the csv containing the labels
        """
        # read the labels csv into a dataframe using pandas
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        # extract the list of labels from the dataframe
        self.labels = [str(label[0]) for label in self.labels_df.values]
        # get the length of the list of labels
        self.no_of_labels = len(self.labels)
        # create a dictionary mapping labels to integers
        self.label_to_int = dict(zip(self.labels, range(self.no_of_labels)))
        # create a dictionary mapping integers to labels
        self.int_to_label = dict(enumerate(self.labels))

    def load_video_labels(self, path_subset, mode="label"):
        """ Loads a Dataframe from a csv
        # Arguments
            path_subset : String, path to the csv to load
            mode        : String, (default: label), if mode is set to "label", filters rows if the labels exists in the labels Dataframe loaded previously
        # Returns
            A DataFrame
        """
        if mode=="input":
            names=['video_id']
        elif mode=="label":
            names=['video_id', 'label']
        
        df = pd.read_csv(path_subset, sep=';', names=names) 
        
        if mode == "label":
            df = df[df.label.isin(self.labels)]

        return df
    
    def categorical_to_label(self, vector):
        """ Convert a vector to its associated string label
        # Arguments
            vector : Vector representing the label of a video
        # Returns
            Returns a String that is the label of a video
        """
        return self.int_to_label[np.where(vector==1)[0][0]]