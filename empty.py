import os
remo_count=0
temp = []


txt_names = os.listdir("./data/test/labels/") # dir of txt files
for x in txt_names:
    with open('./data/test/labels/' + x, 'r') as file:  # openeing txt files
        lines = file.readlines()
        if len(lines) == 0: # if thee txt file is empty
            file_name = os.path.splitext(x)[0] # save file name without suffic
            temp.append('./data/test/labels/'+ x) # add the txt file name to a list cus it is still open
            print('./data/test/labels/'+ x)
            try:
                os.remove('./data/test/images/'+ file_name + '.jpg') # remove the img file that corresonds to the txt file
            except:
                print('Could not find the jpg version')
            remo_count += 1
file.close()

for i in temp:

    os.remove(i) # remove all empty txt files


print(remo_count)


                    
