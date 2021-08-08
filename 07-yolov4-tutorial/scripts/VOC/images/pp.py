# -*- coding: utf-8 -*-   
import os 
  
def file_name(file_dir):  
  li=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
      if os.path.splitext(file)[1] == '.jpg': 
        li.append(file) 
  return li
with open('s.txt','w+') as f:
   a = file_name('./val')
   for i in a:
       f.write(str(i)+' 0'+'\n')
   f.close()
