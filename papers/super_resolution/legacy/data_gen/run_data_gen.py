import json
import ast
import sys

# method to call data_create_mp / gan_data_create_mp
def invoke_generator(ds , split) :

	img_type = sys.argv[1] ;

	if ds["type"] == "synth" :
		import gen_images as data_gen
	else :
		import gan_data_create_mp as data_gen

	for font in ds["data"] :

		print('Font', font , ds["data"][font]["name"]) ;

		for file in ds["data"][font]["files"] :
			print(file) ;
			words = ast.literal_eval(open(ds["src_path"] + '/' + file ,'r').read()) ;	
			wc = int(split * len(words)) ;

			if ds["type"] == "synth" :
				words = words[:wc] ;
			else :
				#print wc ;
				words = words[len(words) - wc:] ;
			#print len(words) ;
			data_gen.main("fonts/" + ds["data"][font]["name"], words, file, ds["dest_path"] ,ds["type"], img_type) ;


		'''
		sp = len(words) / 38
		# temp code start
		for i in range(38) :
			st = i * sp ;
			end = ( i + 1 ) * sp ;
			data_gen.main(ds["all_fonts"]["name"], words, st,end,i ,file, ds["dest_path"] ,ds["type"]) ;
		'''

def main() :

	ds = {} ;
	with open("dataset.json","r") as f :
		ds = json.load(f) ;

	# synth : gan ratio for word list
	total = int(ds["gan_split"]) + int(ds["synth_split"]);

	# image generation
	invoke_generator(ds,float(ds["synth_split"])/total) ;


main() ;
