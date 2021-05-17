import os, os.path, re, csv, glob,itertools, pandas
import lxml.etree as ET
import pandas as pd
from csv import DictWriter,writer

#namespace
tei_ns = {"tei" : "http://www.tei-c.org/ns/1.0"}

#path of directory
dir_path = os.path.dirname(os.path.realpath(__file__))

#Import places and coordinates we are going to look for
colnames = ['name', 'coord']
data = pd.read_csv(os.path.join(dir_path,'listPlaces','propres.tsv'), names=colnames, delimiter='\t')
#list of places
placeNames = data.name.tolist()
#dictionary of places+coords
placeDict = data.set_index('name')['coord'].to_dict()

# create tsv file to store result
listPlaces = "listPlaces.tsv"
with open(os.path.join(dir_path,'listPlaces',listPlaces), 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['code','author','title', 'genre', 'subgenre','place'])

# create tsv file to store result with duplicates
listPlaces_simple = "listPlaces_simple.tsv"
with open(os.path.join(dir_path,'listPlaces',listPlaces_simple), 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['code','author','title', 'genre', 'subgenre','place'])

# add row to tsv file
def append_list_as_row(list_of_elem):
    # Open file in append mode
    with open(os.path.join(dir_path,'listPlaces',"listPlaces.tsv"), 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj, delimiter='\t')
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# add row to tsv file
def append_list_as_row_simple(list_of_elem):
    # Open file in append mode
    with open(os.path.join(dir_path,'listPlaces',"listPlaces_simple.tsv"), 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj, delimiter='\t')
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

#loop over files in Data directory, where plays are stored in xml
for filename in os.listdir(os.path.join(dir_path,'data')):
	print(os.path.join(dir_path,'data',filename))
	file = open(os.path.join(dir_path,'data',filename), "r")
	#basename of filename
	basename = os.path.splitext(filename)[0]
	#get metadata out of xml
	xml = ET.parse(file)
	genre= xml.xpath("//tei:term[@type='genre']/@subtype", namespaces = tei_ns)[0]
	subgenre= xml.xpath("//tei:term[@type='subgenre']/@subtype", namespaces = tei_ns)[0]
	title= xml.xpath("//tei:title[parent::tei:titleStmt]/text()", namespaces = tei_ns)[0]
	author= xml.xpath("//tei:author[parent::tei:titleStmt]/text()", namespaces = tei_ns)[0]
	#look for places in file
	file = open(os.path.join(dir_path,'data',filename), "r")
	data = file.read()
	#we keep only the three main genres
	if subgenre == "tragedy" or subgenre == "comedy" or subgenre == "tragicomedy":
		#loop over all importerd place names
		for placeName in placeNames:
			occurrences = data.count(placeName)
			row_contents = [basename,author,title,genre,subgenre,placeName,occurrences]
			if occurrences > 0:
				append_list_as_row_simple(row_contents)
			#Save result in tsv, repeating each row each according to the number of occurrences
			for _ in itertools.repeat(None, occurrences):
				append_list_as_row(row_contents)

#open list we have just created
colnames = ['basename','author','title','genre','subgenre','placeName','occurrences']
data2 = pd.read_csv(os.path.join(dir_path,'listPlaces','listPlaces.tsv'), names=colnames, delimiter='\t', skiprows=1)
#join with the dictionary of name+coordinates
data2['coord'] = data2['placeName'].map(placeDict)
#save this new result
data2.to_csv(os.path.join(dir_path,'listPlaces','listPlaces_plus.tsv'), sep='\t', encoding='utf-8')

data3 = pd.read_csv(os.path.join(dir_path,'listPlaces','listPlaces_simple.tsv'), names=colnames, delimiter='\t', skiprows=1)
#join with the dictionary of name+coordinates
data3['coord'] = data3['placeName'].map(placeDict)
#save this new result
data3.to_csv(os.path.join(dir_path,'listPlaces','listPlaces_plus_simple.tsv'), sep='\t', encoding='utf-8')
