import glob
import pandas as pd

extension = 'txt'

# Combine training data into one csv file

for file in glob.iglob('imdb_data\\train\\pos\\*.txt'):
    with open(file, encoding='utf8') as file_content:
        review = file_content.read().replace('\n', '')
        combined_pos = pd.concat([pd.DataFrame([review])])

combined_pos['target'] = 1  # pos files so set the target to 1
combined_pos.to_csv('train_reviews.csv', index=False)
print("Done with pos reviews")

for file in glob.iglob('imdb_data\\train\\neg\\*.txt'):
    with open(file, encoding='utf8') as file_content:
        review = file_content.read().replace('\n', '')
        combined_neg = pd.concat([pd.DataFrame([review])])

combined_neg['target'] = -1  # neg files so set the target to -1
combined_neg.to_csv('train_reviews.csv', index=False, mode='a')
print("Done with neg reviews")

