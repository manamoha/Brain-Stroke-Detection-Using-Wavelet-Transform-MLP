*img_cleaning* folder is for data extraction and cleaning from the data set. 

*codes* folder is where main codes are stored.

Data in *cleaned_imgs* are used for trainig/testing

For testing new data you are to modify its path in the *WaveletTransorm.py*, the following command:
- **image_path = os.path.join(root_dir, "ischemic", "test.jpg")**

where *root_dir* is *cleaned_imgs* folder.