from process_data import process_csv_data

train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv', 10, 5)
epoch_x = train_data[0]
print(epoch_x[0].shape)

