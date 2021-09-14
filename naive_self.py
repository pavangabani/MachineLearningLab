import pandas as pd
data ={ 'outlook':['Rainy','Rainy', 'Overcast','Sunny','Sunny','Sunny','Overcast']
        ,'playgolf':['No','No','Yes','Yes','Yes','No','Yes']}
y=0
n=0
for i in range(len(data['playgolf'])):
    if data['playgolf'][i]=='Yes':
        y=y+1
    else:
        n=n+1

total=y+n
predict=input("Write Input :")

py=0
pn=0
for i in range(len(data['outlook'])):
    if data['outlook'][i]==predict and data['playgolf'][i]=='Yes':
        py=py+1
    if data['outlook'][i]==predict and data['playgolf'][i]=='No':
        pn=pn+1

p_rainy_given_y = py/y
p_rainy_given_n= pn/n

print("probablity_of_Input_given_NO : ",p_rainy_given_n,"\nprobablity_of_Input_given_Yes : ",p_rainy_given_y)

f_p_YES=p_rainy_given_y*y/total
f_p_NO=p_rainy_given_n*n/total

if f_p_NO<f_p_YES:
    print("prediction : YES")
else:
    print("prediction : NO")





        