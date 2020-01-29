from src.Processor import Processor

adult = "./datasets/adult/adult.data"

aReader = Processor(adult)

for index, row in aReader.raw.head().iterrows():
    print(row[0])