from random import randrange
import numpy as np

class RandomBatchGenerator(object):

    def __init__(self, data, num_steps, num_padding, attr_column, target_column, batch_size, fill=0.0):
        self.data = data
        self.num_steps = num_steps
        self.num_padding = num_padding
        self.batch_size = batch_size
        self.attr_col = attr_column
        self.target_col = target_column
        self.fill = fill

        self.idx_errors = []


    def generate(self):
        if not self.num_padding:
            x = np.full((self.batch_size, self.num_steps, len(self.attr_col)), self.fill)
            y = np.zeros((self.batch_size, len(self.target_col)))
        else:
            x = np.full((self.batch_size, self.num_steps + self.num_padding, len(self.attr_col)), self.fill)
            y = np.zeros((self.batch_size, self.num_padding, len(self.target_col)))
        while True:
            i = 0
            while i < self.batch_size:
                current_idx = randrange(len(self.data) - self.num_steps - self.num_padding if self.num_padding else 0)
                if current_idx not in range(223234 - self.num_padding - self.num_steps, 223234): #junction of dataset, not ideal but easy for now
                    try:
                        x[i, :self.num_steps, :] = self.data.loc[current_idx:current_idx + self.num_steps - 1, self.attr_col].to_numpy()
                        if not self.num_padding:
                            y[i, :] = self.data.loc[current_idx + self.num_steps, self.target_col].to_numpy()
                        else:
                            #print(self.data.loc[current_idx + self.num_steps:current_idx + self.num_steps + self.num_padding - 1, self.target_col].to_numpy()[0])
                            y[i, :, :] = self.data.loc[current_idx + self.num_steps:current_idx + self.num_steps + self.num_padding - 1, self.target_col].to_numpy()
                    except Exception as e:
                        print("error : " + str(e))
                        self.idx_errors.append(current_idx)
                        exit(0)
                        i = i - 1

                    i = i + 1
            yield x, y