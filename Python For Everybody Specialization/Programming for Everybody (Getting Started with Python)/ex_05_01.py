list =[]

while True:
    pass
    try:
        num = input("Enter a number: ")
        if num.lower() == "done":
            break
        else:
            num = float(num)
            list.append(num)
    except:
        print("Invalid Input")

total = sum(list)
count = len(list)
mean = total/count

print("Sum is {a}. Count is {b}. Average is {c}.".format(a=total,b=count,c=mean))
