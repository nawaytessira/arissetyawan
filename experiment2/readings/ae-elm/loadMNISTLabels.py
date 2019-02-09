# @mfunction("labels")
def loadMNISTLabels(filename=None):
    #loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    #the labels for the MNIST images

    fp = fopen(filename, mstring('rb'))
    _assert(fp != -1, mcat([mstring('Could not open '), filename, mstring('')]))

    magic = fread(fp, 1, mstring('int32'), 0, mstring('ieee-be'))
    _assert(magic == 2049, mcat([mstring('Bad magic number in '), filename, mstring('')]))

    numLabels = fread(fp, 1, mstring('int32'), 0, mstring('ieee-be'))

    labels = fread(fp, inf, mstring('unsigned char'))

    _assert(size(labels, 1) == numLabels, mstring('Mismatch in label count'))

    fclose(fp)

# end
