def make_eyelist(infile):

    import numpy as np

    # List of eyes open
    openeye = '../allopeneyes.txt'
    openeye_out = '../openeyes_list.txt'

    # List of eyes closed
    closedeye = '../allclosedeyes.txt'
    closedeye_out = '../closedeyes_list.txt'

    # Reserve 20% for testing the method later
    usefraction = 0.8

    # Begin reading each image list and constructing the final training lists
    openlist = np.genfromtxt(openeye, dtype=None)
    closedlist = np.genfromtxt(closedeye, dtype=None)

    outopen = open(openeye_out, 'w')
    outclosed = open(closedeye_out, 'w')

    nopen = len(openlist)
    nclosed = len(closedlist)

    nopen_res = int(round(nopen*usefraction))
    nclosed_res = int(round(nopen*usefraction))

    openlist = openlist[0:nopen_res]
    closedlist = closedlist[0:nclosed_res]
    
    for openname in openlist:
        outopen.write(openname + '\n')

    outopen.close()

    for closedname in closedlist:
        outclosed.write(closedname + '\n')

    outclosed.close()
    
    print('OpenEyes:Using %d images out of %d total' % (nopen_res, nopen))
    print('ClosedEyes: Using %d images out of %d total' % (nclosed_res, nclosed))
