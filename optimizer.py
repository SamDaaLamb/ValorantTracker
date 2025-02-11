
import os

count = 0
next = False
saves = list()
directory = input("folder name:")

print(os.listdir('./'+ directory + '/'))
file_names = os.listdir('./'+ directory + '/')
print(os.path.splitext(file_names[1])[1])

file_names.reverse() # so .txt is first


for x in file_names:
    if next == True:
        next = False
        continue
    if os.path.splitext(x)[1] == ".txt":
        next = True
        continue

    count+=1
    os.remove('./'+ directory + '/' + x)

print( "You removed", count, " files")