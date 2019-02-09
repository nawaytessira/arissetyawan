# @mfunction("h, array")
def display_network(A=None, opt_normalize=None, opt_graycolor=None, cols=None, opt_colmajor=None):
    # This function visualizes filters in matrix A. Each column of A is a
    # filter. We will reshape each column into a square image and visualizes
    # on each cell of the visualization panel. 
    # All other parameters are optional, usually you do not need to worry
    # about it.
    # opt_normalize: whether we need to normalize the filter so that all of
    # them can have similar contrast. Default value is true.
    # opt_graycolor: whether we use gray as the heat map. Default is true.
    # cols: how many columns are there in the display. Default value is the
    # squareroot of the number of columns in A.
    # opt_colmajor: you can switch convention to row major for A. In that
    # case, each row of A is a filter. Default value is false.
    warning(mstring('off'), mstring('all'))

    if not exist(mstring('opt_normalize'), mstring('var')) or isempty(opt_normalize):
        opt_normalize = true
    end

    if not exist(mstring('opt_graycolor'), mstring('var')) or isempty(opt_graycolor):
        opt_graycolor = true
    end

    if not exist(mstring('opt_colmajor'), mstring('var')) or isempty(opt_colmajor):
        opt_colmajor = false
    end

    # rescale
    A = A - mean(A(mslice[:]))

    if opt_graycolor:
        colormap(gray)
    end

    # compute rows, cols
    [L, M] = size(A)
    sz = sqrt(L)
    buf = 1
    if not exist(mstring('cols'), mstring('var')):
        if floor(sqrt(M)) ** 2 != M:
            n = ceil(sqrt(M))
            while mod(M, n) != 0 and n < 1.2 * sqrt(M):
                n = n + 1; print n
            end
            m = ceil(M / n)
        else:
            n = sqrt(M)
            m = n
        end
    else:
        n = cols
        m = ceil(M / n)
    end

    array = -ones(buf + m * (sz + buf), buf + n * (sz + buf))

    if not opt_graycolor:
        array = 0.1 *elmul* array
    end


    if not opt_colmajor:
        k = 1
        for i in mslice[1:m]:
            for j in mslice[1:n]:
                if k > M:
                    continue
                end
                clim = max(abs(A(mslice[:], k)))
                if opt_normalize:
                    array(buf + (i - 1) * (sz + buf) + (mslice[1:sz]), buf + (j - 1) * (sz + buf) + (mslice[1:sz])).lvalue = reshape(A(mslice[:], k), sz, sz) / clim
                else:
                    array(buf + (i - 1) * (sz + buf) + (mslice[1:sz]), buf + (j - 1) * (sz + buf) + (mslice[1:sz])).lvalue = reshape(A(mslice[:], k), sz, sz) / max(abs(A(mslice[:])))
                end
                k = k + 1
            end
        end
    else:
        k = 1
        for j in mslice[1:n]:
            for i in mslice[1:m]:
                if k > M:
                    continue
                end
                clim = max(abs(A(mslice[:], k)))
                if opt_normalize:
                    array(buf + (i - 1) * (sz + buf) + (mslice[1:sz]), buf + (j - 1) * (sz + buf) + (mslice[1:sz])).lvalue = reshape(A(mslice[:], k), sz, sz) / clim
                else:
                    array(buf + (i - 1) * (sz + buf) + (mslice[1:sz]), buf + (j - 1) * (sz + buf) + (mslice[1:sz])).lvalue = reshape(A(mslice[:], k), sz, sz)
                end
                k = k + 1
            end
        end
    end

    if opt_graycolor:
        h = imagesc(array, mstring('EraseMode'), mstring('none'), mcat([-1, 1]))
    else:
        h = imagesc(array, mstring('EraseMode'), mstring('none'), mcat([-1, 1]))
    end
    axis(mstring('image'), mstring('off'))

    drawnow()

    warning(mstring('on'), mstring('all'))