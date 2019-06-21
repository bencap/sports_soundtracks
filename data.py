# Ben Capodanno
# CS251, data.py
# This file contains a data class that reads in data from a file
# 02/19/2019 updated 04/02/2019

import numpy as np
import scipy.stats as sps
import sys
import csv
import time
import analysis

class Data:

    def __init__( self, filename = None ):
        # Create and initialize class fields
        self.headers = []
        self.types = []
        self.data = []
        self.enum = {}
        self.header2col = {}
        self.headers_s = []
        self.data_s = []
        self.header2col_s = {}
        self.plotting = []

        # Read File
        if ( filename != None ):
            self.read( filename )

    # output a neat string representation of the (numerical) data set
    def __str__( self, num = True ):
        # set an offset based on the length of the first header

        if not num:
            headers = self.get_headers( True )
            if len( headers ) == 0:
                return "No Data in String Matrix"
            offset = len( headers[0] ) * 3

        else:
            headers = self.get_headers()
            offset = len( headers[0] ) * 2

        # print the first five column heads
        s = headers[0].ljust( offset )
        for i in headers[1:5]:
            s += i.ljust( offset )
        if self.get_num_dimensions( not num ) > 5:
            s += "and " + str( ( self.get_num_dimensions() - 5 ) ) + " other columns"
        print( s )

        # set the numerical offset and check how many rows
        if num:
            offset = "%" + str( offset ) + ".3f"
        rows = self.get_num_points()
        overflow = False

        if rows > 25:
            rows = 25
            overflow = True

        # print the first 25 rows
        for i in range( 0, rows ):
            if num:
                s = str( self.get_value( headers[0], i ) )
                for header in headers[1:5]:

                    s += offset % ( self.get_value( header, i ) )
                print( s )
            else:
                s = str( self.get_value( headers[0], i, True ) ).ljust( offset )
                for header in headers[1:5]:
                    s += self.get_value( header, i, True ).ljust( offset )
                print( s )

        if overflow:
            print( "and " + str( ( self.get_num_points() - 25 ) ) + " other rows." )

        return 1

    # read in a csv file, taking in the important metadata
    def read( self, filename ):
        # open the file with universal read and set up enum iterator and string i
        fp = open( filename, 'rU' )
        tmp_enum = {}
        enum_iterator = 1.0
        string_indices = []

        csv_reader = csv.reader( fp )

        self.headers = next( csv_reader )
        self.types = next( csv_reader )

        # Strip the types of whitespace and find where strings exist
        for i in range( len( self.types ) ):
            self.types[i] = self.types[i].strip()
            if( self.types[i] == "string" ):
                self.headers_s.append( self.headers[i] )
                string_indices.append( i )

        # Loop through each line and decide what to do with each value
        for line in csv_reader:
            toAdd = []
            toAdd_s = []

            for i in range( len( line ) ):
                # if we know the value is a string add it to the string matrix
                if i in string_indices:
                    stripped = line[i].strip()
                    toAdd_s.append( stripped )
                    continue

                # we can append numeric types directly as floats
                if self.types[i] == "numeric":
                    toAdd.append( float( line[i] ) )

                # if the key is already in the dict, pass this func
                # if the key isn't add it with value iterator and increment
                elif self.types[i] == "enum":
                    if line[i] in tmp_enum:
                        pass
                    else:
                        tmp_enum[ line[i] ] = enum_iterator
                        self.enum[enum_iterator] = line[i]
                        enum_iterator = enum_iterator + 1

                    toAdd.append( float( tmp_enum[ line[i] ] ) )

                # Find how many days it has been since jan. 1, 1970
                # IMPLEMENTATION ASSUMPTIONS
                # This implementation counts leap years as one additional day every four years, not a quarter day each year
                # Dates should be in the American date format (MM/DD/YYYY OR MM/DD/YY) and separated by a slash '/'
                # The format YYYYMM or YYYYMMDD is also acceptable and is handled by the catch portion of the statement
                elif self.types[i] == "date":
                    date = 0.0
                    month_length = { 1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

                    # split date into list of individual aspects
                    splitDate = line[i].split( "/" )

                    # try parses two formats, except parses other two
                    try:
                        initial_year = int( splitDate[2] )
                        # find number of leap years (-1) and add the -1 back if past february
                        leaps = -1 + ( ( initial_year - 1968 ) // 4 )
                        if( ( int( splitDate[0] ) > 2 ) and ( leaps > -1 ) ):
                            leaps = leaps + 1

                        # elongate years formatted as YY to subtract better in future
                        if len( splitDate[2] ) < 4:
                            if int( splitDate[2] ) > 69:
                                splitDate[2] = "19" + splitDate[2]
                            else:
                                splitDate[2] = "20" + splitDate[2]

                        # make the dates integers
                        for i in range( len( splitDate ) ):
                            splitDate[i] = int( splitDate[i] )

                        # starting at epoch (1/1/1970), count days caused by years
                        initial = ( splitDate[2] - 1970 ) * 365
                        secondary = initial

                        # starting at epoch (1/1/1970), count days caused by months
                        for i in range( 1, splitDate[0] ):
                            secondary = initial + month_length[i]

                        # starting at epoch (1/1/1970), count days caused by days
                        final_date = secondary + splitDate[1] + leaps

                    except IndexError:
                        initial_year = int( splitDate[0][0:4] )
                        # find number of leap years (-1) and add the -1 back if past february
                        leaps = -1 + ( ( initial_year - 1968 ) // 4 )
                        if( ( initial_year > 2 ) and ( leaps > -1 ) ):
                            leaps = leaps + 1
                        # starting at epoch (1/1/1970), count days caused by years
                        initial = ( initial_year - 1970 ) * 365
                        secondary = -1 + initial
                        # starting at epoch (1/1/1970), count days caused by months
                        for i in range( 1, int( splitDate[0][4:6] ) ):
                            final_date = secondary + month_length[i]
                        # test for YYYYMMDD format
                        try:
                            final_date = final_date + int( splitDate[0][6:8] ) + leaps
                        # otherwise add leap years as extra days
                        except ( IndexError, ValueError ):
                            final_date = final_date + leaps
                    finally:
                        toAdd.append( float( final_date ) )

            self.data.append( toAdd )
            self.data_s.append( toAdd_s )

        # remove headers and string references from main matrix
        for i in reversed( string_indices ):
            self.headers.pop( i )
            self.types.pop( i )

        # Strip whitespace from headers and set up header to column dict
        for i in range( 0, len( self.headers ) ):
            self.headers[i] = self.headers[i].strip()
            self.header2col[ self.headers[i] ] = i
        for i in range( 0, len( self.headers_s ) ):
            self.headers_s[i] = self.headers_s[i].strip()
            self.header2col_s[ self.headers_s[i] ] = i

        self.data = np.matrix( self.data )
        self.data_s = np.matrix( self.data_s )

        return 1

    # return a list of the data headers
    def get_headers( self, s = False ):
        if s:
            return self.headers_s
        return self.headers

    # return a list of the data types
    def get_types( self ):
        return self.types

    # return the number of columns
    def get_num_dimensions( self, s = False ):
        if s:
            return self.data_s.shape[1]
        return self.data.shape[1]

    # return the number of rows
    def get_num_points( self, s = False ):
        if s:
            return self.data_s.shape[0]
        return self.data.shape[0]

    # return a list of all values in a given row
    def get_row( self, index, s = False):
        if s:
            return self.data_s[index]
        return self.data[ index ]

    # takes in a name, type, and matrix of points and adds it to the data
    def add_col( self, name, type, matrix ):
        '''Returns -1 if invalid type
        Returns -(size of matrix) if size of matrix does not match current points
        Returns (size of matrix) if data was inserted successfully'''

        if type != "string" and type != "numeric" and type != "enum" and type != "date":
            raise ValueError('Invalid type: ' + str( name ) + ". Please enter type == 'string', 'numeric', 'enum', or 'date'.")
            return -1
        if type == "string":
            if( np.size( matrix, 0 ) != self.get_num_points( True ) ):
                raise ValueError('Row length of new data must equate to length of current matrix')
                return -np.size( matrix, 0 )
            self.headers_s.append( str( name ) )
            self.data_s = np.hstack( ( self.data_s, matrix ) )
            self.header2col_s[str(name)] = get_num_dimensions( True ) - 1
        else:
            self.headers.append( str( name ) )
            self.types.append( str( type ) )
            self.data = np.hstack( ( self.data, matrix ) )
            self.header2col[str(name)] = self.get_num_dimensions() - 1
        return np.size( matrix, 0 )

    # allow user to access any value in the matrix
    # control flow allows for using either the string name or the index
    def get_value( self, header, index, s = False ):
        # if index use that
        if isinstance( header, int ):
            if s:
                return self.data_s.item( ( index, header ) )
            return self.data.item( ( index, header ) )
        # else check if string is in the header list
        else:
            try:
                if s:
                    return self.data_s.item( ( index, self.header2col_s[ header ] ) )
                return self.data.item( ( index, self.header2col[ header ] ) )
            except KeyError:
                print( "No header (" + header + ") in dataset." )
                return -9999

    # allow user to specify a subset of the matrix to return with a header list
    # this list can be either indices or strings
    # also allows the user to get specific rows, given by a tuple of (lower, upper)
    def subset( self, cols = None, rows = None, s = False ):
        ix = []
        iy = rows

        # Allow for default values to be members of self with this logic
        if cols is None:
            if s:
                cols = self.headers_s
            else:
                cols = self.headers

        ix = analysis.convert_indices( self, cols )

        if rows is None:
            iy = ( 0, self.get_num_points() )

        return np.matrix( self.data[ iy[0]:iy[1], ix ] )

    # write data out to a csv file
    def write( self, filename, headers = None, s = False ):
        # this allows just one control flow for physical writing
        if s:
            temp_head = self.headers
            temp_data = self.data
            temp_2col = self.header2col
            headers = self.get_headers( True )
            self.headers = self.headers_s
            self.data = self.data_s
            self.header2col = self.header2col_s

        if headers == None:
            headers = self.get_headers()

        fn = open( str( filename ), "w" )

        indices = []
        # write headers and types
        i = 0
        for header in headers:
            if i == len( headers ) - 1:
                fn.write( header )
                indices.append( self.header2col[header] )
                i += 1
                continue
            fn.write( header + "," )
            indices.append( self.header2col[header] )
            i += 1

        fn.write("\n")
        i = 0
        for idx in indices:
            if i == len( indices ) - 1:
                fn.write( self.types[idx] )
                continue
            fn.write( self.types[idx] + "," )
            i += 1
        fn.write("\n")

        # write data
        for i in range( np.size( self.data, 0 ) ):
            for j in range( np.size( self.data, 1 ) ):
                if j == ( np.size( self.data, 1 ) - 1 ):
                    fn.write( str( self.data[i,j] ) )
                    continue
                fn.write( str( self.data[i,j] ) + "," )
            fn.write( "\n" )

        fn.close()

        # revert temperary string data
        if s:
            self.headers = temp_head
            self.data = temp_data
            self.header2col = temp_2col

class PCAData( Data ):

    def __init__(self, proj_data, eigenvectors, eigenvalues, mean_dv, orig_dh ):
        super( PCAData, self ).__init__( None )
        self.data = proj_data
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean_dv = mean_dv
        self.orig_dh = orig_dh
        cols = self.get_num_dimensions()
        for i in range( cols ):
            if i < 10:
                self.headers.append( "PCA0" + str( i ) )
            else:
                self.headers.append( "PCA" + str( i ) )
            self.header2col[ self.headers[i] ] = i
            self.types.append( "numeric" )

    def get_eigenvalues( self ):
        return self.eigenvalues

    def get_eigenvectors( self ):
        return self.eigenvectors

    def get_original_means( self ):
        return self.mean_dv

    def get_original_headers( self ):
        return self.orig_dh

class KData( Data ):

    def __init__( self, data, headers, means, codes, errors, quality = None ):
        super( KData, self ).__init__( None )
        self.headers = headers
        for i in range( len( headers ) ):
            print( self.headers )
            self.header2col[ self.headers[i] ] = i
            self.types.append( "numeric" )
        self.data = data
        self.codebook = means
        self.codes = codes
        self.errors = errors
        self.k = np.size( self.codebook, 0 )
        if quality == None:
            self.quality = analysis.kmeans_quality( errors, self.k )
        else: self.quality = quality

    def get_codebook( self ):
        return self.codebook

    def get_codes( self ):
        return self.codes

    def get_errors( self ):
        return self.errors

    def get_k( self ):
        return self.k

    def write( self, filename, headers = None ):
        if headers == None:
            headers = self.get_headers()

        fn = open( str( filename ), "w" )

        fn.write( "Cluster Means - \n")
        for i in range( np.size( self.codebook, 0 ) ):
            fn.write( "Cluster " + str( i ) + ": " + str( self.codebook[i,:] ) + "\n" )

        fn.write( "\nCluster Membership and Errors - \n" )
        for i in range( np.size( self.codes.T, 0 ) ):
            fn.write( "Point " + str( i ) + " -> Cluster " + str( self.codes.T[i] ) + " with error: " + str( self.errors[i] ) + "\n" )

        fn.write("\nOverall Quality Score: " + str( self.quality ) + "\n")

        fn.write( "\nOriginal Data - \n\n")
        indices = []
        # write headers and types
        for header in headers:
            fn.write( header + " " )
            indices.append( self.header2col[header] )
        fn.write("\n")
        for idx in indices:
            fn.write( self.types[idx] + " " )
        fn.write( "\n" )
        # write data
        for i in range( np.size( self.data, 0 ) ):
            for j in range( np.size( self.data, 1 ) ):
                fn.write( str( self.data[i,j] ) + " " )
            fn.write( "\n" )

        fn.close()

if __name__ == "__main__":

    # Data Class Tests
    data = Data( sys.argv[1] )
    print( "Headers" )
    print( data.get_headers(), "\n" )

    print( "String Headers" )
    print( data.get_headers( True ), "\n" )

    print( "Types" )
    print( data.get_types(), "\n" )

    print( "Dimensions" )
    print( data.get_num_dimensions(), "\n" )

    print( "String Dimensions" )
    print( data.get_num_dimensions( True ), "\n" )

    print( "Points" )
    print( data.get_num_points(), "\n" )

    print( "String Points" )
    print( data.get_num_points( True ), "\n" )

    row = 0
    print( "Row Index: %d" % row)
    print( data.get_row( row ), "\n" )

    print( "Row in String Matrix: %d" % row)
    print( data.get_row( row, True ), "\n" )

    row = 1
    col = 0
    col_s = data.headers[0]
    try:
        print( "Specific Data Points")
        print( data.get_value( col, row ) )
        print( data.get_value( col, row ), "\n" )
    except IndexError:
        print( IndexError, " change row and columns in test\n" )

    try:
        print( "Specific Data Points in String")
        print( data.get_value( col, row, True) )
        print( data.get_value( col, row, True ), "\n" )
    except IndexError:
        print( IndexError, " change row and columns in test\n" )

    print( "Numeric Matrix Representation" )
    data.__str__()
    print()
    print( "String Matrix Representation" )
    data.__str__( False )
    print()

    print( "Subsets of Matrix\n" )
    col = data.headers[1:3]
    row = ( 1, 5 )
    print( "All rows with Column Subset" )
    print( data.subset( col ), "\n" )
    print( "All columns with Row Subset" )
    print( data.subset( rows = row ), "\n" )
    print( "Subset Rows and Columns")
    print( data.subset( col, row ), "\n" )

    print( "Range of Numeric Data" )
    print( analysis.data_range( data, data.get_headers() ), "\n" )

    print( "IQR of the Numeric Columns" )
    print( analysis.data_iqr( data, data.get_headers() ), "\n" )

    print( "Mean of the Numeric Columns" )
    print( analysis.data_mean( data, data.get_headers() ), "\n" )

    print( "Median of the Numeric Columns" )
    print( analysis.data_median( data, data.get_headers() ), "\n" )

    print( "StDev of the Numeric Columns" )
    print( analysis.data_stdev( data, data.get_headers() ), "\n" )

    print( "Variance of the Numeric Columns" )
    print( analysis.data_variance( data, data.get_headers() ), "\n" )

    print( "Normalized Numeric Columns" )
    print( analysis.normalize_columns_separately( data, data.get_headers() ), "\n" )

    print( "Normalized Numeric Array" )
    print( analysis.normalize_columns_together( data, data.get_headers() ), "\n" )

    print( "Data with first row added to end" )
    data.add_col( "added data", "numeric", data.subset( cols = [0] ) )
    data.__str__()
    print()

    print( "Writing data to File: test.csv" )
    data.write( "test", data.get_headers() )
