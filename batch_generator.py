import numpy as np

class PandasBatchGenerator(object):

    def __init__(self, data, num_steps, num_padding, attr_column, target_column, batch_size, skip_step):
        self.data = data
        self.num_steps = num_steps
        self.num_padding = num_padding
        self.batch_size = batch_size
        self.attr_col = attr_column
        self.target_col = target_column

        self.current_idx = 0
        self.skip_step = skip_step
        self.idx_errors = []


    def generate(self):
        if not self.num_padding:
            x = np.full((self.batch_size, self.num_steps, len(self.attr_col)), 0.0)
            y = np.zeros((self.batch_size, len(self.target_col)))
        else:
            x = np.full((self.batch_size, self.num_steps + self.num_padding, len(self.attr_col)), 0.0)
            y = np.zeros((self.batch_size, self.num_padding, len(self.target_col)))
        while True:
            i = 0
            while i < self.batch_size:
                if self.current_idx + self.num_steps + (self.num_padding if self.num_padding else 0) >= len(self.data):
                    self.current_idx = 0

                try:
                    x[i, :self.num_steps, :] = self.data.loc[self.current_idx:self.current_idx + self.num_steps - 1, self.attr_col].to_numpy()
                    if not self.num_padding:
                        y[i, :] = self.data.loc[self.current_idx + self.num_steps, self.target_col].to_numpy()
                    else:
                        #print(self.data.loc[self.current_idx + self.num_steps:self.current_idx + self.num_steps + self.num_padding - 1, self.target_col].to_numpy()[0])
                        y[i, :, :] = self.data.loc[self.current_idx + self.num_steps:self.current_idx + self.num_steps + self.num_padding - 1, self.target_col].to_numpy()
                except Exception as e:
                    #      self.target_col].to_numpy())
                    #print(self.target_col)
                    print("error : " + str(e))
                    self.idx_errors.append(self.current_idx)
                    exit(0)
                    i = i - 1
#                print(x[i])
#                print(y[i])

                self.current_idx += self.skip_step
                i = i + 1
            yield x, y