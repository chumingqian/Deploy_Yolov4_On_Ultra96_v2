line = open("calibjz.txt").readlines()
with open('a.txt','w+') as f:
    for i in line:
        f.write(i.split(" ")[0]+'\n')
    f.close()
