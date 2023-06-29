import os

# file name with extension
file_name = os.path.basename(r'\dataset\test\nondemented\nondemented_al4.jpg')

# file name without extension
print(os.path.splitext(file_name)[0])



# initializing string
test_str = os.path.basename(r'C:\Users\91974\OneDrive\Desktop\minor\Minor\dataset\test\nondemented\nondemented_al1')

# printing original string
print(str(test_str))

# loop to iterating characters
temp = 0
for chr in test_str:
	
# checking if character is numeric,
# saving index
    if chr.isdigit():
	    temp = test_str.index(chr)

# printing result
print(str(test_str[0 : temp]))
