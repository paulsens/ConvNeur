for i in range(self.batch_size):
    batch_z[i, :, :] = to_categorical(sample_dec[i][1:], num_classes=vocabulary)