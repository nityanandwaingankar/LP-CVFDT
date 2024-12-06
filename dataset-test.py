from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
monk_s_problems = fetch_ucirepo(id=70) 
  
# data (as pandas dataframes) 
X = monk_s_problems.data.features 
Y = monk_s_problems.data.targets 
  
# metadata 
# print(monk_s_problems.metadata) 
  
# variable information 
# print(monk_s_problems.variables) 
print("X is :")
print(type(X))

print("y is :") 
print(type(Y))



print("Updating tree with streaming data...\n")
# for (x1,x2) in X.iterrows():
#     # print(f"Data Point {i+1}: X={X}, y={y}")
#     print(f"Data Point:{x2.to_numpy()}")
#     # for i in range(len(x2)):
#     #     print(y[i])
#     print("\n")


for y1,y2 in Y.iterrows():
    # tree.update(x, y)
    # print("updated ateleast once")
    print((y2[0]))

# tree.print_tree()



# print("\nMaking predictions on new data:")
# for x,y in zip(X,Y):
#     # prediction = tree.predict(x)
#     print(f"Test Point {+1}: X={x} => Prediction: {prediction}")    