def read_iq_file(filename, dtype=np.float32,channels=2):

  raw_data = np.fromfile(filename, dtype=dtype)
  iq_data = raw_data[::channels} + 1j*raw_data[1::channels}

  return iq_data
