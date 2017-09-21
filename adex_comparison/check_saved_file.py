import tables


filename = 'quantities_spectral_elnath.h5'
h5file = tables.open_file(filename,mode='r')
root = h5file.root
test = h5file.root.lambda_diffusive

print(test.shape)