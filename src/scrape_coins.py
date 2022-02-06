'''
Created on 6 Feb 2022

@author: maltaweel
'''
import os
from os import listdir
import csv
import requests
from bs4 import BeautifulSoup 

#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

def loadData():
    
    def loadData(self, start, end):
        
        #open the file(s) in the modified directory
        try:
            for f in listdir(directory):
                
                #should only be .csv files
                if '.csv' not in f:
                    continue
                
             
                #open individual files
                with open(os.path.join(directory,f),'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    #read the rows of data
                    for row in reader:
                        uri=row['URI']
                        title=row['Title']
                        rec_id=row['RecordId']
                        from_date=row['From Date']
                        to_date=row['To Date']
                        authority=row['Authority']
        
        except IOError:
            print ("Could not read file:", csvfile)

    
    def openLink(uri):
        r = requests.get(uri)
        soup = BeautifulSoup(r.content)
        
        #scrape text
        soup.find_all("a")
        # scrape images
        links = soup.findAll('img')

def main():
    loadData()
    print('run')


if __name__ == '__main__':
    main()