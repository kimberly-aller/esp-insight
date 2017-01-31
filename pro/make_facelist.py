import pdb

def make_facelist(infile):
    '''
    Makes the list of the open eyes on full faces where it reserves
    20% of the images for testing.
    
    Setup to run with make_face_data to make the arrays for the Keras
    NN input for training and validation.

    Mod History: 
    2017-01-30 - Created from make_eyelist

    '''
    import numpy as np

    # List of eyes open
    openeye = '../allopenfaces_list.txt'
    openeye_out = '../openfaces_list.txt'
    openeye_val_out = '../openfaces_val_list.txt'

    # List of eyes closed
    closedeye = '../allclosedfaces_list.txt'
    closedeye_out = '../closedfaces_list.txt'
    closedeye_val_out = '../closedfaces_val_list.txt'

    # Reserve 20% for testing the method later
    usefraction = 0.8

    # Begin reading each image list and constructing the final training lists
    openlist = np.genfromtxt(openeye, dtype=None)
    closedlist = np.genfromtxt(closedeye, dtype=None)

    # Open your files to print output lists
    outopen = open(openeye_out, 'w')
    outopen_val = open(openeye_val_out, 'w')
    outclosed = open(closedeye_out, 'w')
    outclosed_val = open(closedeye_val_out, 'w')

    # Find the 80%/20% faces for each open/closed
    nopen = len(openlist)
    nclosed = len(closedlist)

    nopen_res = int(round(nopen*usefraction))
    nclosed_res = int(round(nopen*usefraction))

    openlist_train = openlist[0:nopen_res]
    openlist_val = openlist[nopen_res:]
    closedlist_train = closedlist[0:nclosed_res]
    closedlist_val = closedlist[nclosed_res:]

    # Make the open eye face arrays: 80% for train and 20% for val
    for openname in openlist_train:
        outopen.write(openname + '\n')
    outopen.close()
    for openname in openlist_val:
        outopen_val.write(openname + '\n')
    outopen_val.close()
    pdb.set_trace()

    # Make the closed eye face arrays: 80% for train and 20% for val
    for closedname in closedlist_train:
        outclosed.write(closedname + '\n')
    outclosed.close()
    for closedname in closedlist_val:
        outclosed_val.write(closedname + '\n')
    outclosed_val.close()

    # Print the numbers keeping for training
    print('OpenFaces:Using %d images out of %d total' % (nopen_res, nopen))
    print('ClosedFaces: Using %d images out of %d total' % (nclosed_res, nclosed))
