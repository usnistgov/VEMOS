# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:48:56 2017

@author: Eve Fleisig

This class represents one element of a data set for the PatternAnalyzer GUI.

"""
from collections import OrderedDict

class DataRecord:
    """Represents one element of a data set for the PatternAnalyzer GUI.
    
    Attributes
    ----------
    id : str
        The ID of the record.
    groups : list of str
        Lists the groups to which the record belongs.
    matches : list of str
        Lists the IDs of the records that match this record.
    files : OrderedDict() of {str : str}
        Dict of file types and file names; e.g.:
        {"Image": "record1_img.jpg","Segmentation": "record1_seg.png",...}
        
    """
    
    def __init__(self, new_id, groups, matches, files):
        
        self.id = str(new_id)
        self.groups = groups
        self.matches = matches
        self.files = OrderedDict(sorted(files.items()))
        
         
    def print_data(self):
        """Prints the record's ID, groups, matches, and files.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        """
        print("\nID:", self.id, "\nGroups:", self.groups)
        print("Matches:", self.matches, "\nFiles:", self.files, "\n")
    
     
    def get_description_file_format(self):
        """Returns the record's attributes in description file format.
        
        Groups and matches are returned in the format (group1, group2,...)
        All other elements are returned separated by semicolons.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        str
            The record's attributes, in description file format.
            
        """
        
        formatted_groups = "(" + ", ".join(self.groups) + ")"
        formatted_matches = "(" + ", ".join(self.matches) + ")"
        elements = [self.id, formatted_groups, formatted_matches]
        elements.extend([key + ": " + self.files[key] for key in self.files])
        return '; '.join(elements) + '\n'
        