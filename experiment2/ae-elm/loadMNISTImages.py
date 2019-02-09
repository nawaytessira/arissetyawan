# @mfunction("images")
def loadMNISTImages(filename=None):
    #loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    #the raw MNIST images

    fp = fopen(filename, mstring('rb'))
    _assert(fp != -1, mcat([mstring('Could not open '), filename, mstring('')]))

    magic = fread(fp, 1, mstring('int32'), 0, mstring('ieee-be'))
    _assert(magic == 2051, mcat([mstring('Bad magic number in '), filename, mstring('')]))

    numImages = fread(fp, 1, mstring('int32'), 0, mstring('ieee-be'))
    numRows = fread(fp, 1, mstring('int32'), 0, mstring('ieee-be'))
    numCols = fread(fp, 1, mstring('int32'), 0, mstring('ieee-be'))

    images = fread(fp, inf, mstring('unsigned char'))
    images = reshape(images, numCols, numRows, numImages)
    images = permute(images, mcat([2, 1, 3]))

    fclose(fp)

    # Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3))
    # Convert to double and rescale to [0,1]
    images = double(images) / 255
    # images = (double(images) / 127.5) - 1;
# end