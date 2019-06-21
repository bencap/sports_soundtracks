""" analyze.py


This file contains functions
    *
    *

Created by Ben Capodanno on June 19th, 2019. Updated June 19th, 2019.
"""

import data
import analysis
import sys

def main( filename ):
    songs = data.Data( filename )
    pca_songs = analysis.pca( songs, songs.get_headers() )

if __name__ == "__main__":
    if len( sys.argv ) < 2:
        print( "USAGE: python analyze.py <filename>" )
        exit()
    main( sys.argv[1] )