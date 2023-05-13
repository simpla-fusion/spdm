# Definition of the class structures in file imas.py
import imas
import numpy
import sys
import os

'''
This sample program will create a pulse file (shot 13, run 1) and will
put an example of equilibirium IDS using put_slice methods.
'''

# This routine reads an array of pfsystems IDSs in the database, filling
# some fields of the IDSS

TEST_DATABASE_NAME = 'test'


def put_ids():
    """Class Itm is the main class for the UAL.

    It contains a set of field classes, each corresponding to a IDS
    defined in the UAL The parameters passed to this creator define the
    shot and run number. The second pair of arguments defines the
    reference shot and run and is used when the a new database is
    created, as in this example.

    """

    shot = 13
    time = 1
    interp = 1

    imas_obj = imas.ids(13, 1, 13, 1)
    # Create a new instance of database
    imas_obj.create_env("fydev", "test", "3")

    if imas_obj.isConnected():
        print('Creation of data entry OK!')
    else:
        print('Creation of data entry FAILED!')
        sys.exit()

    number = 10

    # Allocate a first generic vector and its time base
    lentime_1 = 3
    vect1DDouble_1 = numpy.empty([lentime_1])
    time_1 = numpy.empty([lentime_1])

    for i in range(lentime_1):
        time_1[i] = i
        vect1DDouble_1[i] = i * 10

    print('========================================================')
    print(time_1)
    print(vect1DDouble_1)

    # Allocate a second generic vector and its time base
    lentime_2 = 4
    vect1DDouble_2 = numpy.empty([lentime_2])
    time_2 = numpy.empty([lentime_2])

    for i in range(lentime_2):
        time_2[i] = i
        vect1DDouble_2[i] = i * 11

    print('========================================================')
    print(time_2)
    print(vect1DDouble_2)

    vect2DDouble_1 = numpy.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            vect2DDouble_1[i, j] = i * 100 + j

    print('========================================================')
    print(vect2DDouble_1)
    # Allocate a second generic vector and its time base
    lentime_2 = 4
    vect1DDouble_2 = numpy.empty([lentime_2])
    time_2 = numpy.empty([lentime_2])

    for i in range(lentime_2):
        time_2[i] = i
        vect1DDouble_2[i] = i * 11

    print('========================================================')
    print(time_2)
    print(vect1DDouble_2)

    vect2DDouble_1 = numpy.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            vect2DDouble_1[i, j] = i * 100 + j

    print('========================================================')
    print(vect2DDouble_1)

    vect2DDouble_2 = vect2DDouble_1 + 10000
    '''
    print( '========================================================')
    print( vect3DDouble_2)
    '''
    imas_obj.equilibrium.ids_properties.comment = 'This is a test ids'

    # A sample int

    # Mandatory to define this property
    imas_obj.equilibrium.ids_properties.homogeneous_time = 1
    imas_obj.equilibrium.resize(1)
    imas_obj.equilibrium.time_slice[0].profiles_2d.resize(2)
    imas_obj.equilibrium.time_slice[0].profiles_2d[0].Mesh_type.name = 'Mesh TYPE 1A'
    imas_obj.equilibrium.time_slice[0].profiles_2d[1].Mesh_type.name = 'Mesh TYPE 2B'

    imas_obj.equilibrium.time.resize(1)
    imas_obj.equilibrium.time_slice[0].profiles_2d[0].r.resize(3, 3)
    imas_obj.equilibrium.time_slice[0].profiles_2d[1].r.resize(3, 3)

    print('Start Put, writing first slice')
    imas_obj.equilibrium.time_slice[0].profiles_2d[0].r[:, 0] = vect2DDouble_1[0, :]
    imas_obj.equilibrium.time_slice[0].profiles_2d[1].r[:, 0] = vect2DDouble_2[0, :]
    imas_obj.equilibrium.time[0] = time_1[0]
    imas_obj.equilibrium.put()
    print('Completed Put ')

    for i in range(lentime_1):
        print('========================================================')
        print('vect3DDouble_1[i,:,:]')
        print(vect2DDouble_1[i, :])
        print('========================================================')

        imas_obj.equilibrium.time_slice[0].profiles_2d[0].r[:, i] = vect2DDouble_1[i, :]
        print('========================================================')
        print('imas_obj.equilibrium.time_slice[0].profiles_2d[0].r')
        print(imas_obj.equilibrium.time_slice[0].profiles_2d[0].r[:, i])
        print('========================================================')
        imas_obj.equilibrium.time_slice[0].profiles_2d[1].r[:, i] = vect2DDouble_2[i, :]
        imas_obj.equilibrium.time[0] = time_1[i]
        print(('Writing slice={0}'.format(i)))
        imas_obj.equilibrium.putSlice()

    print('========================================================')
    print(imas_obj.equilibrium.time_slice[0].profiles_2d[0].r)

    '''
        print( '========================================================')
        print (imas_obj.equilibrium.time_slice[0].profiles_2d[1].r)
        '''
    imas_obj.close()


put_ids()
