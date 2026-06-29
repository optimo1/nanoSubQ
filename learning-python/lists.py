list = [1,2,3,4,5]

list.append(2)
list.extend([6,7,8])
list.insert(0,10)

list.remove(3)
a = list.pop(4)
#list.clear()

list.reverse()
list.sort()
print(list.count(2))

print(a,list)