from process_data import process_csv_data

train_data, test_data = process_csv_data('../data/train.csv', '../data/test.csv', 10, 5)
epoch_x, epoch_y = train_data[0], train_data[1]

print(epoch_x)