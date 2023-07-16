import training_dataset as td

X,Y = (td.read_images('dataset\\train',32))
print(type(X))
print(X)
