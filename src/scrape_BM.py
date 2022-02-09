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

#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

filename=os.path.join(pn,'output','output_bm.csv')
fieldnames=['id','title','image']

def loadData(directory, f):
    #open individual files
    with open(os.path.join(directory,f),'r') as csvfile:
        reader = csv.DictReader(csvfile)
                
        with open(filename, 'w') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames,delimiter=',', quotechar='"')
            writer.writeheader()   
                    
            #read the rows of data
            for row in reader:
                
                image=row['Image']
                objType=row['Object type']
                mNumber=row['Museum number']
                description=row['Description']
                
                img_folder=os.path.join(pn,'images_bm')
                downloadImage(image,'',img_folder)
                

def downloadImage(img,name2,folder):
    
    download_img = urllib.request.urlopen(img)
    name=img.split('/')
    path_to_data=os.path.join(folder,name2+name[len(name)-1])
    
    txt = open(path_to_data, "wb")
    
    #write the binary data
    txt.write(download_img.read())
        
def main():
    uri='https://www.britishmuseum.org/collection/search?keyword=hadrian&keyword=coins'
    loadData(uri)
    print('run')


if __name__ == '__main__':
    main()