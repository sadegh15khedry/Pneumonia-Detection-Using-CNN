from sklearn.model_selection import train_test_split
import os

# def load_data(file_name):
#     return pd.read_csv('../datasets/'+file_name)

# def split_data(df, feature_column, label_column, test_size=.2, random_state=50):
#     # x = df.drop(columns= [feature_column])
#     # y = df.drop(columns= [label_column])
#     return train_test_split(df[feature_column], df[label_column], test_size=test_size, random_state=random_state)

# def preprocess_data(df):
#     df = df.dropna()
#     df.drop_duplicates(keep=False) 
#     return df 

def rename_images(image_dir, new_name_prefix, counting_number=0):
    for image_file in os.listdir(image_dir):
        old_image_path = os.path.join(image_dir, image_file)
        new_image_path = os.path.join(image_dir, new_name_prefix +'_'+ str(counting_number) + '.jpeg')
        os.rename(old_image_path, new_image_path)
        counting_number += 1