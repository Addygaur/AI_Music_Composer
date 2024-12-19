import pandas as pd

def merge_final():

    # Load datasets
    main_data = pd.read_csv("emopia_features.csv")
    key_data = pd.read_csv("key_features.csv")

    # Merge on filename or similar identifier
    enhanced_data = pd.merge(main_data, key_data, left_on="source_filename", right_on="filename", how="left")

    enhanced_data.drop(columns=['filename'], inplace=True)

    # Save enhanced dataset
    enhanced_data.to_csv("emopia_features.csv", index=False)

    print("Updated Dataframe")
