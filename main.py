from features import features_extract
from merge import merge
from major_minor import process_mid
from merge_2 import merge_final
from classification import process_data

def dataset_extract():
    import muspy
    emopia = muspy.EMOPIADataset("data/emopia/", download_and_extract=True) #Dataset in data/emopia folder
    emopia.convert()
    music = emopia[0]
    print(music.annotations[0].annotation)
 

if __name__ == "__main__":
    #To download the emopia dataset and store the metadata about the mid files in the converted folder
    dataset_extract()

    #extract all the relevant features from the json files and save it in a csv file
    features_extract.traverse_files()

    #merge the tempo values from key_mode_tempo csv file
    merge()

    #Extract the major and minor key ratio for mid files
    process_mid()

    #merge the major and minor values
    merge_final()

    #Train the classification model
    process_data()













