'''
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

context = ssl._create_unverified_context()



#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

filename=os.path.join(pn,'output','output_bm.csv')
fieldnames=['id','title','denomination','image']

def loadData():
    #open individual files
    for f in listdir(directory):
                
            #should only be .csv files
        if 'collections' not in f:
            continue
    
        with open(os.path.join(directory,f),'r') as csvfile:
            reader = csv.DictReader(csvfile)
        
        
            with open(filename, 'w') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=fieldnames,delimiter=',', quotechar='"')
                writer.writeheader()   
                
                
                #read the rows of data
                for row in reader:
                    try:
                        image=row['Image']
                        #objType=row['Object type']
                        mNumber=row['Museum number']
                        description=row['Description']
                        denomination=row['Denomination']
                
                        img_folder=os.path.join(pn,'images_bm')
                        downloadImage(image,'',img_folder)
                    
                        writeOutput(writer,description,denomination,mNumber,image.split('/'))
                        csvf.flush()
                    
                    except ValueError as e:
                        print(image,e)
                        continue
                        
                

def downloadImage(img,name2,folder):
    
    download_img = urlopen(img,context=context)
    name=img.split('/')
    name=name[len(name)-1].replace('preview_','')
    path_to_data=os.path.join(folder,name2+name)
    
    txt = open(path_to_data, "wb")
    
    #write the binary data
    txt.write(download_img.read())
   
    
def writeOutput(writer,content,denomination,idd, l2):
    image_name=l2[len(l2)-1]
    image_name=image_name.replace('preview_','')
    writer.writerow({'id': idd,'title':content,'denomination':denomination,'image':image_name})
    
        
def main():
   
    loadData()
    print('run')


if __name__ == '__main__':
    main()