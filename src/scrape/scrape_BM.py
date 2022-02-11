'''
Module and class to scrape the BM's coin data using
a data file in the data folder (i.e., collections.csv type file).
Created on 9 Feb 2022

@author: maltaweel
'''

import os
from os import listdir
import csv
import sys
import time
from urllib.request import urlopen
from bs4 import BeautifulSoup
import certifi
import ssl
from scrape.scrape_coins import Scrape

context = ssl._create_unverified_context()



#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

#output path for csv
filename=os.path.join(pn,'output','output_bm.csv')

#fieldnames to print out in the csv output
fieldnames=['id','description','denomination','image']

class BM_Scrape(Scrape):
    
    '''
    Method to load data from /data/ folder
    '''
    def loadData(self):
        
        #open individual files
        for f in listdir(directory):
                
            #should only be .csv files and have collections in the title
            if 'collections' not in f:
                continue
            
            #open the output file and write it
            with open(filename, 'w') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=fieldnames,delimiter=',', quotechar='"')
                writer.writeheader()  
                
                #read the file 
                with open(os.path.join(directory,f),'r') as csvfile:
                    reader = csv.DictReader(csvfile) 
                        #read the rows of data
                    for row in reader:
                        try:
                            
                            #get the image link
                            image=row['Image']
                            #objType=row['Object type']
                            
                            #museum number
                            mNumber=row['Museum number']
                            
                            #description
                            description=row['Description']
                            
                            #coin denomination
                            denomination=row['Denomination']
                
                            #create the output folder to put it in
                            img_folder=os.path.join(pn,'images_bm')
                            
                            #download the image
                            self.downloadImage(image,'',img_folder)
                    
                            #write out the descriptive data
                            self.writeOutput(writer,description,denomination,mNumber,image.split('/'))
                            
                            #flush the output to the file
                            csvf.flush()
                    
                        except Exception as e:
                            print(e)
                            continue
    
    '''
    Method for downloading images from a given link.
    @param img- image data
    @param name2- second name to add to the image name
    @param folder- the folder to put the image in
    '''
    def downloadImage(self,img,name2,folder):
    
        #the image to download is downloaded here
        download_img = urlopen(img,context=context)
        
        #split the name; create the name for the file
        name=img.split('/')
        name=name[len(name)-1].replace('preview_','')
        path_to_data=os.path.join(folder,name2+name)
    
        #open the image data
        txt = open(path_to_data, "wb")
    
        #write the binary data
        txt.write(download_img.read())
    
    '''
    Method to scrape information from links within a site (uri).
    @param uri- the uri link to get data from
    @param folder- the folder to put data in
    @param denomination- the denomination of the coin type
    @param writer- the writer to write results
    @param csvf- the csv file to write the outputs
    '''
    def writeOutput(self, writer,content,denomination,idd, l2):
        #get the name by removing the web address and keeping the last part
        image_name=l2[len(l2)-1]
        
        #replace the preview_ part
        image_name=image_name.replace('preview_','')
        
        #write the data
        writer.writerow({'id': idd,'description':content,'denomination':denomination,'image':image_name})
    
'''
The main to run the module.
'''        
def main():
    scrap_bm=BM_Scrape()
    scrap_bm.loadData()
    print('run')


if __name__ == '__main__':
    main()