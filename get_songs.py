""" get_songs.py

This file runs command line processes to get the Spotify IDs of 
all songs in the list curated

This file contains functions
    * authorize() - Use unique user key to get stdout containing a Spotify API access code
    * search( sp, track, lim ) - Searches Spotify for a given tracks id, features, and analysis objects
    * fix_str( string, fixers ) - Removes substrings( elements of the list 'fixers' ) from the a string
    * format_outstream( track_name, track_id, features, analysis, f_keys, a_keys ) - prepares gathered data
                                                                                     for csv writing
    * main( filename ) - Execute above and other control flow. This files main function

Created by Ben Capodanno on June 17th, 2019. Updated June 19th, 2019
"""

import sys
import csv
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def authorize():
    """ creates a object proving user authentification

    :returns: A Spotipy object containing user authentification
    :rtype: Spotipy Spotify object
    """

    client_credentials_manager = SpotifyClientCredentials( os.environ['SPOTIPY-CLIENT-ID'], os.environ['SPOTIPY-CLIENT-SECRET'] )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def search( sp, track, lim=1 ):
    """ searches API for id and features of a song

    :param sp: A spotipy Spotify object proving authentification
    :type sp: Spotipy Spotify object
    :param track: The name of the track being searched
    :type track: String
    :param limit: The number of songs to return - this might only work with limit = 1
    :type limit: Int
    :returns: The unique ID, Features, and Low-Level characteristics of the song
    :rtype: Tuple
    """

    identifier = sp.search( q="track: " + track, limit=lim, type="track" )['tracks']['items'][0]['id']
    features   = sp.audio_features( identifier )
    analisys   = sp.audio_analysis( identifier )

    return identifier, features, analisys

def fix_str( string, fixers ):
    """ Given a list of strings, this function removes them from a string

    :param string: The string being fixed
    :type string: String
    :param fixers: The strings to remove from the list
    :type fixers: List
    :returns: The string with substrings removed
    :rtype: String
    """

    for item in fixers:
        string = string.replace( item, "" )
    return string

def format_outstream( track_name, game, track_id, features, analysis, f_keys, a_keys ):
    """ parses input to return a string that can be written to a file

    :param track_name: The name of the track being written
    :type track: String
    :param game: ie class. The game whose soundtrack the song is a member
    :type game: String
    :param track_id: Spotify ID of the track being written
    :type track_id: String
    :param features: Echo Nest statistics (high level characteristics)
    :type features: Dictionary
    :param analysis: Spotify song analysis characteristics (low level characteristics)
    :type analysis: Dictionary
    :param f_keys: A list of the audio feature dict keys
    :type f_keys: List
    :param a_keys: A list of the audio analysis dict keys
    :type a_keys: List 
    :returns: neecessary info in csv format
    :rtype: String
    """

    # "track_name", "track_id", "danceability", "energy", "key", "loudness", "mode", "speechiness", 
    # "acousticness", "instrumentalness", "liveness", "valence", "tempo", "avg_bar_len", "avg_beat_len", 
    # **"sections"**, **"segments"**, "avg_tatum_len", "song_len", **"additional track features here"**
    out_string = track_name + "," + track_id + "," + game + ","
    
    # for each of the keys in the dict that features or analysis returns, get some data.
    # This data is always of the same form in the feature set, but not in the analysis set
    for key in f_keys: out_string += str( features[0][key] ) + ","
    for key in a_keys:
        if key is "bars" or key is "beats" or key is "tatums":
            sum = 0
            for dictionary in analysis[key]:
                sum += dictionary[ "duration" ]
            out_string += str( sum / len( analysis[key] ) ) + ","
        if key is "sections": pass #TODO: Retrieve meaningful data from Sections
        if key is "segments": pass #TODO: Retrieve meaningful data from Segments
        if key is "track": out_string += str( analysis[key]["duration"]) + "\n"
                
    return out_string

def main( filename ):
    """ this files main function, call above to get a list of unique IDs of songs in csv file

    :param filename: The name of the file containing songs
    :type filename: String
    :returns: None
    :rtype: None
    """

    f_keys = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
    a_keys = ["bars", "beats", "sections", "segments", "tatums", "track"]
    lines = 0
    search_err = 0

    # open file for reading and skip heading line
    fn = open( filename, "r" )
    out = open( "id_file.csv", "w" )
    csv_reader = csv.reader( fn )
    out.write( "track_name,unique_id,game,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,avg_bar_len,avg_beat_len,avg_tatum_len,song_len\n" )
    next( csv_reader )

    # call authorize to enable API calls and increased rate
    sp = authorize()

    # result is top result for each search in song db as returned by spotify API
    # this is a dictionary which is parsed and assigned to identifier.
    for line in csv_reader:
        lines += 1
        # try to get unique IDs and features. Except case covers errors in API accesses
        # removing certain characters may benefit search. Removing """ and "," aids CSV generation
        try:
            line[0] = fix_str( line[0], ["- ", "!", "&", "(", ")"] )
            identifier, features, analysis = search( sp, line[0], 1)
            line[0] = fix_str( line[0], ["\"", ","] )
            out.write( format_outstream( line[0], line[3], identifier, features, analysis, f_keys, a_keys ) )
        except IndexError:
            print( "Search Error on line %d. Could not find ID for: %s." % ( lines, line[0] ) )
            search_err += 1

        if lines % 250 == 0: print( "Accessed through line %d" % lines )
    
    print( "API search failures on %d / %d lines." % (search_err, lines ) )

    fn.close()
    out.close()

if __name__ == "__main__":
    if len( sys.argv ) < 2:
        print( "USAGE: python get_songs.py <filename>")
        exit()
    main( sys.argv[1] )