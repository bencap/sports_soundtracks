""" convertnba.py

This file converts web txt data to csv usable format

This file contains functions:

    * delimit( filename ) - writes song names and artists from nba in to a more accessible format
    * main( filename ) - run above, this programs main function
    *

Created by Ben Capodanno on June 12, 2019. Updated June 17, 2019
"""
import sys

def delimit( filename ):
    """ Writes song names and artists to an output file in a standard format

    :param filename: The filename of the input
    :type filename: String
    :returns: None
    :rtype: None
    """

    inp = open( filename, "r" )
    out = open( "nbaout.csv", "w" )
    for line in inp:
        line = line.strip("\n").replace(',', '')
        next = line.split("-")
        alt = line.split("–")
        alt2 = line.split("—")
        # separators are one of three characters, here handle each case
        try:
            out.write( "None," + next[0] + "," + next[1] + ",None,NBA\n" )
        except IndexError:
            try:
                out.write( "None," + alt[0] + "," + alt[1] + ",None,NBA\n" )
            except IndexError:
                out.write( "None," + alt2[0] + "," + alt2[1] + ",None,NBA\n" )
    inp.close()
    out.close()

def main( filename ):
    """ This programs main function, runs delimit

    :param filename: The filename of the input file
    :type filename: String
    :returns: None
    :rtype: None
    """

    delimit( filename )

if __name__ == "__main__":
    if len( sys.argv ) < 2:
        print( "USAGE: python convertnba.py <filename>" )
        exit()
    main( sys.argv[1] )
