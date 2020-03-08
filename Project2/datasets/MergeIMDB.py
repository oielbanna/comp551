"""
Instead of uploading the whole dataset to GitHub with the data separated in different files,
this script was used to merge the data from the data files into one csv for each subset (train/test) while tagging the
reviews (positive 1 or negative -1)
"""
import glob
import pandas as pd

# Pick which subset to merge its data
subset = 'test'

# Combine training data into one csv file
pos_reviews = []
for file in glob.iglob('imdb_data\\{}\\pos\\*.txt'.format(subset)):
    with open(file, encoding='utf8') as file_content:
        review = file_content.read().replace('\n', '')
        pos_reviews.append([review])

df = pd.DataFrame(pos_reviews, columns=['reviews'])
df['target'] = 1  # pos files so set the target to 1
df.to_csv('{}_reviews.csv'.format(subset), index=False)
print("Done with pos reviews")

neg_reviews = []
for file in glob.iglob('imdb_data\\{}\\neg\\*.txt'.format(subset)):
    with open(file, encoding='utf8') as file_content:
        review = file_content.read().replace('\n', '')
        neg_reviews.append([review])

df = pd.DataFrame(neg_reviews, columns=['reviews'])
df['target'] = -1  # neg files so set the target to -1
df.to_csv('{}_reviews.csv'.format(subset), index=False, header=False, mode='a')
print("Done with neg reviews")

