"""
Shuffle raw data, and export them to shuffle_data.dat.
"""
import random

def main():
    file_x = open("ex4x.dat", 'r')
    file_y = open("ex4y.dat", 'r')
    shuffle_data = open('shuffle_data.dat', 'w')

    raw_data = []

    temp_x = file_x.readline() 
    while temp_x != '':
        temp_x = temp_x.lstrip()
        temp_x = temp_x.rstrip('\n')
        raw_data.append(temp_x)
        temp_x = file_x.readline()

    temp_y = file_y.readline()
    i = 0
    while temp_y != '':
        raw_data[i] = raw_data[i] + temp_y
        temp_y = file_y.readline()
        i += 1

    random.shuffle(raw_data)
    for i in raw_data:
        shuffle_data.writelines(i)

if __name__ == '__main__':
    main()