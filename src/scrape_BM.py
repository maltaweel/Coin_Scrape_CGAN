'''
Created on 9 Feb 2022

@author: maltaweel
'''

import os
from os import listdir
import csv
import sys
import time
import urllib.request
from bs4 import BeautifulSoup 

def loadData(uri):
    #get request uri
    soup = BeautifulSoup(urllib.request.urlopen(uri), "html.parser")
        
    # scrape images
    links = soup.findAll('a',{'target':'_blank'})
    
    #iterate through the links for images   
        
      
    for linkz in links:
          
            
        ids=linkz['href']
        
def main():
    uri='https://www.britishmuseum.org/collection/search?keyword=hadrian&keyword=coins'
    loadData(uri)
    print('run')


if __name__ == '__main__':
    main()