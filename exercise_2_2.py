import os
file_path = '/rds/user/jdb206/hpc-work/MLMI2/exp/phone_map'

# initialize an empty dictionary
d = {}

# open the file and read the lines
with open(file_path, 'r') as f:
    lines = f.readlines()

# iterate over the lines
for line in lines:
    # split the line into key and value
    key, value = line.strip().split(':')
    # add the key-value pair to the dictionary
    d[key] = value



output_path = os.path.join(os.getcwd(), 'vocab_39.txt')

with open(output_path, 'w') as f:
    f.writelines("%s\n" % item for item in set(d.values()))

assert len(set(d.values())) == 40

#after this I manually switch the empty line to _ and then move to first line after this