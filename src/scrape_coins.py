'''
Created on 6 Feb 2022

@author: maltaweel
'''
import os
from os import listdir
import csv
import sys
import time
import urllib.request
from bs4 import BeautifulSoup 
from rpy2.tests.robjects.test_dataframe import test_from_csvfile

#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

filename=os.path.join(pn,'output','output.csv')
fieldnames=['id','title','image']

csvfile=''
def loadData():

    #open the file(s) in the modified directory
    try:
        for f in listdir(directory):
                
            #should only be .csv files
            if '.csv' not in f:
                continue
                
        
            #open individual files
            with open(os.path.join(directory,f),'r') as csvfile:
                reader = csv.DictReader(csvfile)
                
                with open(filename, 'w') as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=fieldnames,delimiter=',', quotechar='"')
                    writer.writeheader()   
                    
                    #read the rows of data
                    for row in reader:
                   
                        uri=row['URI']
                        title=row['Title']
                        rec_id=row['RecordId']
                        from_date=row['From Date']
                        to_date=row['To Date']
                        authority=row['Authority']
                    
                        image_folder=os.path.join(pn,'images',rec_id)

                        openLink(uri,image_folder, writer)
                  
        
    except IOError:
        print ("Could not read file:", csvfile)

def downloadImage(img,name2,folder):
    
    download_img = urllib.request.urlopen(img)
    name=img.split('/')
    path_to_data=os.path.join(folder,name2+name[len(name)-1])
    
    txt = open(path_to_data, "wb")
    
    #write the binary data
    txt.write(download_img.read())
    
def openLink(uri,folder, writer):
    
        #get request uri
        soup = BeautifulSoup(urllib.request.urlopen(uri), "html.parser")
        
        # scrape images
        links = soup.findAll('a',{'target':'_blank'})
    
        #iterate through the links for images   
        
      
        for linkz in links:
          
            
            ids=linkz['href']
        
            if 'http://nomisma.org/' in ids:
                continue
        
            if '"http://numismatics.org/ocre/id/' in ids:
                continue  
            #get the link that is in src (i.e., an image from html)
            try:
                print(ids)
                soup2 = BeautifulSoup(urllib.request.urlopen(ids), "html.parser")
            
                if 'http://numismatics.org' in ids:
                    imgs=soup2.findAll(title='Full resolution image')
                
                    title=soup2.find('meta',property='og:title')
                
                    idd=ids.split('/collection/')[1].strip()
                    
                    for im in imgs:
                        l2=im['href']
                        content=title['content']
                        try:
                            os.mkdir(folder)
                        except OSError:
                            pass
                        downloadImage(l2,'',folder)
                        writeOutput(writer,content,idd,l2.split('/'))
                
                else:
                    doLink(writer,ids,folder)
            except:
                continue
            
       

def doLink(writer,ids,folder):
    
    if 'www.ikmk.at' or 'https://www.univie.ac.at' or 'https://ikmk.smb.museum' in ids:
        soup2 = BeautifulSoup(urllib.request.urlopen(ids), "html.parser")
        obs=soup2.find('img',{'id':'main-image'})
        
        title=soup2.find('title')
        idd=ids.split('object?id=')[1]
        front=obs['src']
    
        revs=front.replace('vs_exp.jpg','rs_opt.jpg')
        
        nl=front.split(idd+'/')[1]
        n1=nl.replace('vs_exp.jpg',idd+'vs_exp.jpg')
        n2=nl.replace('vs_exp.jpg',idd+'rs_opt.jpg')
        
        downloadImage(front,idd,folder)
        downloadImage(revs,idd,folder)
        
        writeOutput(writer,title.contents[0],idd,n1.split('/'))
        writeOutput(writer,title.contents[0],idd,n2.split('/'))
       
    elif 'https://finds.org.uk/database/artefacts/' in ids:
        doPAS(writer,ids,folder)


def doPAS(writer,ids,folder):
    soup2 = BeautifulSoup(urllib.request.urlopen(ids), "html.parser")
    obs=soup2.findAll('img')
    idt=soup2.find('span',{'class':'fourfigure'})
    
    idd=idt.contents[0]
    title=soup2.find('meta',{'property':'og:title'})
    titl=title['content']
   
    src=''
    for i in obs:
        src=i['src']
        
        if '/medium' in src:
            downloadImage(src,'',folder)
            break
        
    writeOutput(writer,titl,idd,src.split('/'))
  
     
     
          
def writeOutput(writer,content,idd, l2):
    image_name=l2[len(l2)-1]
    writer.writerow({'id': idd,'title':content,'image':image_name})
    csvfile.flush()
    

def main():
    loadData()
    print('run')


if __name__ == '__main__':
    main()