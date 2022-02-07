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

#the path to the data folder
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]  
directory=os.path.join(pn,'data')

filename=os.path.join(pn,'output','output.csv')
fieldnames=['id','title','image']

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
                    
                #read the rows of data
                for row in reader:
                    uri=row['URI']
                    title=row['Title']
                    rec_id=row['RecordId']
                    from_date=row['From Date']
                    to_date=row['To Date']
                    authority=row['Authority']
                    
                    image_folder=os.path.join(pn,'images',rec_id)

                    openLink(uri,image_folder)
        
    except IOError:
        print ("Could not read file:", csvfile)

def downloadImage(img,name2,folder):
    
    download_img = urllib.request.urlopen(img)
    name=img.split('/')
    path_to_data=os.path.join(folder,name2+name[len(name)-1])
    
    txt = open(path_to_data, "wb")
    
    #write the binary data
    txt.write(download_img.read())
    
def openLink(uri,folder):
    
    with open(filename, 'w') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        #get request uri
        soup = BeautifulSoup(urllib.request.urlopen(uri), "html.parser")
        
        # scrape images
        links = soup.findAll('a',{'target':'_blank'})
    
        #iterate through the links for images   
        for linkz in links:
         
            id=linkz['href']
        
            if 'http://nomisma.org/' in id:
                continue
        
            if '"http://numismatics.org/ocre/id/' in id:
                continue  
            #get the link that is in src (i.e., an image from html)
            try:
                soup2 = BeautifulSoup(urllib.request.urlopen(id), "html.parser")
            
                if 'http://numismatics.org' in id:
                    imgs=soup2.findAll(title='Full resolution image')
                
                    title=soup2.find('meta',property='og:title')
                
                    idd=id.split('/collection/')[1].strip()
                    
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
                    doLink(writer,id,folder)
            except:
                continue
        
        csvf.close()

def doLink(writer,id,folder):
    if 'www.ikmk.at' in id:
        soup2 = BeautifulSoup(urllib.request.urlopen(id), "html.parser")
        obs=soup2.find('img',{'id':'main-image'})
        
        title=soup2.find('title')
        idd=id.split('object?id=')[1]
        front=obs['src']
    
        revs=front.replace('vs_exp.jpg','rs_opt.jpg')
        
        nl=front.split(idd+'/')[1]
        n1=nl.replace('vs_exp.jpg',idd+'vs_exp.jpg')
        n2=nl.replace('vs_exp.jpg',idd+'rs_opt.jpg')
        
        downloadImage(front,idd,folder)
        downloadImage(revs,idd,folder)
        
        writeOutput(writer,title.contents[0],idd,n1.split('/'))
        writeOutput(writer,title.contents[0],idd,n2.split('/'))
            
def writeOutput(writer,content,idd, l2):
    image_name=l2[len(l2)-1]
    writer.writerow({'id': idd,'title':content,'image':image_name}) 
    

def main():
    loadData()
    print('run')


if __name__ == '__main__':
    main()