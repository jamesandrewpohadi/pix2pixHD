#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""

#from synthesize import synthesize
import os 
import pandas as pd
import csv
from config import Config
from test_one import infer
from shutil import copyfile
from train_mirror import train

class image:
    #input:
    #path to image to be shown"
    def __init__(self, path):
        self.path=path
    def show(self):    
        #show image
        os.system(self.path)

class model(Config):
    def __init__(self):
        Config.__init__(self)
        
    def infer(self, image_label_path, image_inst_path):
        #infer image
        #input:
        #image_label_path: path to image with "_labelIds" ending
        #image_label_path: path to image with "_instanceIds" ending
        if not os.path.exists(self.samples):
            print("creating {}".format(self.samples))
            os.mkdir(self.samples)
        image_name = os.path.basename(image_label_path)[:-13]
        pathh = 'Examples.csv'
        if not os.path.exists(pathh):
            df=pd.DataFrame(columns=["image_name"])
            df.to_csv ("Examples.csv", encoding = "utf-8") 
        else:
            f = open(pathh,'rb')
            df = pd.read_csv(f, index_col=0, encoding = "utf-8")
            df = pd.DataFrame(df)
        row_num = df.iloc[:,0].size+1
        df.loc[row_num] = [image_name]
        df.to_csv ("./Examples.csv", encoding = "utf-8")
        infer(row_num, image_label_path, image_inst_path)
        File = open(os.path.join('transcript.csv'), 'a+') 
        writeCSV = csv.writer(File)
        row = []
        con = ("inference_%.5d" % (row_num))+'|'+image_name
        row.append(con)
        writeCSV.writerow(row)
        input_label  = "./samples/inference_%.5d_input_label.jpg" % (row_num)
        synthesized_image  = "./samples/inference_%.5d_synthesized_image.jpg" % (row_num)
        if os.path.exists(input_label):
            if os.path.exists(synthesized_image):
                return (image(input_label), image(synthesized_image))
            else:
                print("{} doesn't exist".format(synthesized_image))
        else:
            print("{} doesn't exist".format(input_label))
   

    def collect(self, recorder_path = None):
        if recorder_path == None:
            recorder_path = self.recorder_path
        if not os.path.exists(recorder_path):
            print ('No such path: {}\nTry to specify other path by model.collected(recorder_path=PATH_NAME)'.format(recorder_path))
            return 
        if not os.path.exists(self.collected_path):
            print('Creating collected path at {}'.format(self.collected_path))
            os.mkdir(self.collected_path)
            #os.mkdir(os.path.join(self.collected_path,'png'))
            os.mkdir(os.path.join(self.collected_path,'label'))
            os.mkdir(os.path.join(self.collected_path,'inst'))
            os.mkdir(os.path.join(self.collected_path,'img'))
            df=pd.DataFrame(columns=["Text"])
            df.to_csv (os.path.join(self.collected_path,'Index.csv'), encoding = "utf-8") 
        
        a = open(os.path.join(self.collected_path,'Index.csv'), "r")
        row_num  = len(a.readlines())
        a.close()
     
        errorFile = open(os.path.join(self.collected_path,'Index.csv'), 'a+', newline='') 
        writeCSV = csv.writer(errorFile)
     
        label_files = os.listdir(os.path.join(recorder_path,'label'))
        inst_files = os.listdir(os.path.join(recorder_path,'inst'))
        img_files = os.listdir(os.path.join(recorder_path,'img'))
            
        for fi in label_files:       
            if fi.endswith("_label.png") and fi[:-10]+'_inst.png' in inst_files and fi[:-10]+'_img.png' in img_files:
                
                #row = []
                name = ('MY_%.5d' % row_num)
                meaning = fi[:-13]
                #row.append(name)
                #row.append(meaning)
                row = ['{} -> {}'.format(name, meaning)]
                print('{} -> {}'.format(meaning, name)) 
                writeCSV.writerow(row)
                os.rename(os.path.join(recorder_path, 'label', fi),os.path.join(self.collected_path, 'label', name+"_label.png"))
                os.rename(os.path.join(recorder_path, 'inst', fi[:-10]+"_inst.png"),os.path.join(self.collected_path, 'inst', name+"_inst.png"))
                os.rename(os.path.join(recorder_path, 'img', fi[:-10]+"_img.png"),os.path.join(self.collected_path, 'img', name+"_img.png"))
                row_num+=1
        print('done!')
        
    def reframe(self, test_path=None):
        #reframe dataset to reframed_dataset
        #reframed_dataset consist of former_dataset with collected_dataset
        #cp_path = os.path.join(self.reframed_dataset,'png')
        cp_path = self.reframed_dataset
        if test_path == None:
            test_path = self.test_path
        if not os.path.exists(self.reframed_dataset):
            print('creating {}'.format(self.reframed_dataset))
            os.mkdir(self.reframed_dataset)
        if not os.path.exists(cp_path):
            print('creating {} ...'.format(cp_path))
            os.mkdir(cp_path)
        if not os.path.exists(os.path.join(cp_path, 'inst')):
            print('creating {} ...'.format(cp_path, 'inst'))
            os.mkdir(os.path.join(cp_path, 'inst'))
            for image in os.listdir("former_dataset/train_inst"):
                copyfile(os.path.join("former_dataset/train_inst",image),os.path.join(self.reframed_dataset,'inst', os.path.basename(image)))
        if not os.path.exists(os.path.join(cp_path, 'label')):
            print('creating {} ...'.format(cp_path, 'label'))
            os.mkdir(os.path.join(cp_path, 'label'))
            for image in os.listdir("former_dataset/train_label"):
                copyfile(os.path.join("former_dataset/train_label",image),os.path.join(self.reframed_dataset,'label', os.path.basename(image)))
        if not os.path.exists(os.path.join(cp_path, 'img')):
            print('creating {} ...'.format(cp_path, 'img'))
            os.mkdir(os.path.join(cp_path, 'img'))
            for image in os.listdir("former_dataset/train_img"):
                copyfile(os.path.join("former_dataset/train_img",image),os.path.join(self.reframed_dataset,'img', os.path.basename(image)))
        if 'reframe.csv' not in os.listdir(self.reframed_dataset):
            print('creating reframe.csv')
            df=pd.DataFrame()
            df.to_csv ('{}/reframe.csv'.format(self.reframed_dataset), encoding = "utf-8")
        
        a = open(os.path.join(self.reframed_dataset,'reframe.csv'), "r", newline='')
        row_num = (a.readlines())
        a.close()
    
        File = open(os.path.join(self.reframed_dataset,'reframe.csv'), 'a+', newline='') 
        writeCSV = csv.writer(File)
        if not os.path.exists(os.path.join(self.collected_path,'Index.csv')):
            print('Data not collected! Try to run model.collect()!')

        else:       
            a = open(os.path.join(self.collected_path,'Index.csv'), "r")
            a.readline()
            content = a.readlines()
            for line in content:
                tmp = line.split(' -> ')
                inst_name = tmp[0] + '_inst.png'
                label_name = tmp[0] + '_label.png'
                img_name = tmp[0] + '_img.png'
                
                meaning = tmp[1][:-1]
                if not (os.path.exists(os.path.join(cp_path, 'label', label_name)) and os.path.exists(os.path.join(cp_path, 'inst', inst_name))):
                    print('reframing {} ...'.format(tmp[0]))
                    row = []
                    row.append('{} -> {}'.format(tmp[0],meaning))

                    writeCSV.writerow(row)
                    copyfile(os.path.join(self.collected_path, 'label', label_name), os.path.join(cp_path, 'label', label_name))
                    copyfile(os.path.join(self.collected_path, 'inst', inst_name), os.path.join(cp_path, 'inst', inst_name))
                    copyfile(os.path.join(self.collected_path, 'img', img_name), os.path.join(cp_path, 'img', img_name))

        print('done!')
         
    def train(self, name="label2city_512p"):
        #train dataset
        if not os.path.exist(self.reframed_dataset):
            print('./reframed_dataset does not exist, try to use model.reframe()')
            return
        train(name)
