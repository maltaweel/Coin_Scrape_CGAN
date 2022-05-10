'''
Module to scrape data from several different sites (not BM).
This uses input in the data folder (query.csv).
The data include coin images and descriptive information about the coins,
including description and id.

Created on 6 Feb 2022

@author: maltaweel
'''
import os
from os import listdir
import csv
import urllib.request
from bs4 import BeautifulSoup 
#from rpy2.tests.robjects.test_dataframe import test_from_csvfile

#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

#file output path
filename=os.path.join(pn,'output','output.csv')

#fieldnames for output file
fieldnames=['id','description','denomination','image']

#csv file to output
csvfile=''


class Scrape():
    
    '''
    Method to load data from /data/ folder
    '''
    def loadData(self):

        #open the file(s) in the modified directory
        try:
            for f in listdir(directory):
                
                #should only be .csv files
                if '.csv' not in f or 'collections' in f:
                    continue
                
                #this will output data to the output folder
                with open(filename, 'w') as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=fieldnames,delimiter=',', quotechar='"')
                    writer.writeheader()   
                    
                    #open individual files to read from /data folder
                    with open(os.path.join(directory,f),'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                    
                        #read the rows of data
                        for row in reader:
                            
                            #get uri
                            uri=row['URI']
                            #title=row['Title']
                            
                            #record data (id)
                            rec_id=row['RecordId']
                            
                            #coine info
                            denomination=row['Denomination']
                            print(rec_id)
                            #from_date=row['From Date']
                            #to_date=row['To Date']
                            #authority=row['Authority']
                    
                            #info for where the coin images are stored
                            image_folder=os.path.join(pn,'images',rec_id)
                            
                            #now scrape the information
                            self.openLink(uri,image_folder,denomination, writer,csvf)
                        
                  
        
        except IOError as e:
            print ("Could not read file:", csvfile, e)

    '''
    Method for downloading images from a given link.
    @param img- image data
    @param name2- second name to add to the image name
    @param folder- the folder to put the image in
    '''
    def downloadImage(self,img,name2,folder):
        
        #the image to download is downloaded here
        download_img = urllib.request.urlopen(img)
        
        #to make the name split the url link and get the last part of the url
        name=img.split('/')
        path_to_data=os.path.join(folder,name2+name[len(name)-1])
        
        #open the path to the data stream
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
    def openLink(self, uri,folder, denomination,writer, csvf):
    
        #get request uri
        soup = BeautifulSoup(urllib.request.urlopen(uri), "html.parser")
        
        # scrape images
        links = soup.findAll('a',{'target':'_blank'})
    
        #iterate through the links for images   
        for linkz in links:
          
            #get links
            ids=linkz['href']
        
            #skip unneeded links
            if 'http://nomisma.org/' in ids:
                continue
        
            if 'http://numismatics.org/ocre/id/' in ids:
                continue  
            
            #get the link info relevant
            try:
                pathT=os.path.exists(folder)
                
                if pathT!=True:
                    os.mkdir(folder)
                #get numismatic data
                if 'http://numismatics.org' in ids:
                    #use beautifulsoupt to scrape data from link (ids)
                    soup2 = BeautifulSoup(urllib.request.urlopen(ids), "html.parser")
                    #the image info
                    imgs=soup2.findAll(title='Full resolution image')
                    
                    #title info
                    title=soup2.find('meta',property='og:title')
                
                    #get data from the collection link
                    idd=ids.split('/collection/')[1].strip()
                    
                    #get content information from the image
                    for im in imgs:
                        l2=im['href']
                        content=title['content']
                        #write out the image
                        self.downloadImage(l2,'',folder)
                        
                        #wtrite out descriptive data about the image
                        self.writeOutput(writer,content,denomination,idd,l2.split('/'))
                
                else:
                    #if it is not from numismatics.org then see what it is
                    self.doLink(writer,ids,denomination,folder)
            except Exception as e:
                print(e)
                continue
            
            csvf.flush()

    '''
    Method to scrape data from sites that are not numismatics.org but 
    other coin sites.
    @param writer- the write to write data
    @param ids- the id data for the image
    @param denomination- the denomination of the coin
    @param folder- the folder for the data to download in
    '''
    def doLink(self,writer,ids,denomination,folder):
    
        #get ikmk, univie, or other ikmk data
        if 'www.ikmk.at'in ids or 'https://www.univie.ac.at' in ids or 'https://ikmk.smb.museum' in ids:
            #use beuatiful soup to scrape
            soup2 = BeautifulSoup(urllib.request.urlopen(ids), "html.parser")
            obs=soup2.find('img',{'id':'main-image'})
        
            #get descriptive data
            title=soup2.find('title')
            idd=ids.split('object?id=')[1]
            front=obs['src']
    
            #make some replacements for the data name
            revs=front.replace('vs_exp.jpg','rs_opt.jpg')
        
            #split up the image to make the relevant image name to download
            nl=front.split(idd+'/')[1]
            n1=nl.replace('vs_exp.jpg',idd+'vs_exp.jpg')
            n2=nl.replace('vs_exp.jpg',idd+'rs_opt.jpg')
        
            #download images (front and back of coin)
            self.downloadImage(front,idd,folder)
            self.downloadImage(revs,idd,folder)
        
            #write the descriptive data (front and back of coin data)
            self.writeOutput(writer,title.contents[0],denomination,idd,n1.split('/'))
            self.writeOutput(writer,title.contents[0],denomination,idd,n2.split('/'))
       
        #get PAS data
        elif 'https://finds.org.uk/database/artefacts/' in ids:
            self.doPAS(writer,ids,denomination,folder)
    

    '''
    Method for scraping PAS data.
    @param writer- the writer to write data
    @param ids- the id for the images and info to scrape
    @param denomiation- the denomination of the coin
    @param folder- the folder to put the data in
    '''
    def doPAS(self,writer,ids,denomination,folder):
        #beautifulsoup to scrape data
        soup2 = BeautifulSoup(urllib.request.urlopen(ids), "html.parser")
        
        #get image data
        obs=soup2.findAll('img')
        idt=soup2.find('span',{'class':'fourfigure'})
        
        #get content info.
        idd=idt.contents[0]
        title=soup2.find('meta',{'property':'og:title'})
        titl=title['content']
   
        src=''
        for i in obs:
            src=i['src']
        
            #get the right image
            if '/medium' in src:
                self.downloadImage(src,'',folder)
                break
        
        #write descriptive output
        self.writeOutput(writer,titl,denomination,idd,src.split('/'))
  
    '''
    Method to write data to csv file.
    @param writer- the writer
    @param content- the descriptive data of the coin
    @param denomiation- the coin denomination
    @param idd- the id of the coin
    @param l2- the coin file name
    '''
    def writeOutput(self,writer,content,denomination,idd, l2):
        image_name=l2[len(l2)-1]
        writer.writerow({'id': idd,'description':content,'denomination':denomination,'image':image_name})
    
'''
Main method to launch the class to scrape data from several different sites
found in /output/output.csv
'''
def main():
    scrape=Scrape()
    scrape.loadData()
    print('run')


if __name__ == '__main__':
    main()